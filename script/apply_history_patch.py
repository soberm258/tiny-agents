import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


ROOT_MARKERS = [
    r"tiny-agents\\",
    r"tiny-agents/",
]


def _normalize_repo_path(p: str) -> str:
    p = (p or "").strip()
    p = p.replace("\\", "/")
    lower = p.lower()
    idx = lower.find("tiny-agents/")
    if idx >= 0:
        p = p[idx + len("tiny-agents/") :]
    p = p.lstrip("/")
    return p


@dataclass
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: List[str]


@dataclass
class FilePatch:
    old_path: str
    new_path: str
    new_file: bool = False
    deleted_file: bool = False
    hunks: List[Hunk] = None


HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def _read_text_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    data = path.read_text(encoding="utf-8", errors="replace")
    return data.splitlines()


def _write_text_lines(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    # 习惯性保留末尾换行，避免某些工具提示 “No newline at end of file”
    if text and not text.endswith("\n"):
        text += "\n"
    path.write_text(text, encoding="utf-8", errors="strict")


def _apply_hunks(original: List[str], hunks: List[Hunk], file_path: str) -> List[str]:
    out = original[:]
    # 使用指针在 out 上原地应用；hunk 的 old_start 是 1-based
    # 注意：我们严格按上下文匹配，避免误改
    for hunk in hunks:
        # 将 old_start 转为 0-based 索引
        idx = max(0, hunk.old_start - 1)
        # 为了更稳妥，允许在 idx 附近少量漂移（不同换行风格/空行可能导致轻微偏移）
        window = 50
        found = False
        candidate_starts = [idx] + [i for i in range(max(0, idx - window), min(len(out), idx + window + 1)) if i != idx]

        for start in candidate_starts:
            tmp = out[:]
            cur = start
            ok = True
            for raw in hunk.lines:
                if raw.startswith("\\ No newline at end of file"):
                    continue
                if not raw:
                    continue
                tag = raw[0]
                content = raw[1:]
                if tag == " ":
                    if cur >= len(tmp) or tmp[cur] != content:
                        ok = False
                        break
                    cur += 1
                elif tag == "-":
                    if cur >= len(tmp) or tmp[cur] != content:
                        ok = False
                        break
                    del tmp[cur]
                elif tag == "+":
                    tmp.insert(cur, content)
                    cur += 1
                else:
                    ok = False
                    break
            if ok:
                out = tmp
                found = True
                break

        if not found:
            raise RuntimeError(f"补丁应用失败（上下文不匹配）：{file_path} @@ -{hunk.old_start},{hunk.old_count} +{hunk.new_start},{hunk.new_count} @@")
    return out


def parse_history_diff(text: str) -> List[FilePatch]:
    lines = text.splitlines(True)
    patches: List[FilePatch] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.startswith("diff --git "):
            i += 1
            continue
        parts = line.strip().split()
        if len(parts) < 4:
            i += 1
            continue
        old_path = parts[2].removeprefix("a/")
        new_path = parts[3].removeprefix("b/")
        fp = FilePatch(old_path=old_path, new_path=new_path, hunks=[])
        i += 1

        # 读取文件级元信息直到遇到 ---/+++ 或 @@
        while i < len(lines):
            meta = lines[i]
            if meta.startswith("new file mode"):
                fp.new_file = True
            elif meta.startswith("deleted file mode"):
                fp.deleted_file = True
            if meta.startswith("--- ") or meta.startswith("+++ ") or meta.startswith("@@ "):
                break
            i += 1

        # 跳过 ---/+++ 行
        if i < len(lines) and lines[i].startswith("--- "):
            i += 1
        if i < len(lines) and lines[i].startswith("+++ "):
            i += 1

        # 读取 hunks
        while i < len(lines):
            if lines[i].startswith("diff --git "):
                break
            m = HUNK_RE.match(lines[i])
            if not m:
                i += 1
                continue
            old_start = int(m.group(1))
            old_count = int(m.group(2) or "1")
            new_start = int(m.group(3))
            new_count = int(m.group(4) or "1")
            i += 1
            hunk_lines: List[str] = []
            while i < len(lines):
                if lines[i].startswith("diff --git ") or HUNK_RE.match(lines[i]):
                    break
                hunk_lines.append(lines[i].rstrip("\n").rstrip("\r"))
                i += 1
            fp.hunks.append(
                Hunk(
                    old_start=old_start,
                    old_count=old_count,
                    new_start=new_start,
                    new_count=new_count,
                    lines=hunk_lines,
                )
            )

        patches.append(fp)
    return patches


def backup_paths(paths: List[Path], backup_root: Path) -> None:
    for p in paths:
        if not p.exists():
            continue
        dest = backup_root / p.as_posix()
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)


def main() -> None:
    repo_root = Path(".").resolve()
    history_path = repo_root / "history.txt"
    if not history_path.exists():
        raise SystemExit("找不到 history.txt（请放在仓库根目录）。")

    text = history_path.read_text(encoding="utf-8", errors="replace")
    patches = parse_history_diff(text)
    if not patches:
        raise SystemExit("history.txt 中没有解析到 diff --git 段落。")

    rel_paths: List[Path] = []
    for fp in patches:
        rp = _normalize_repo_path(fp.new_path)
        if rp:
            rel_paths.append(Path(rp))

    # 备份：仅备份会被写入的路径
    ts = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = repo_root / "recovery_backup" / f"history_restore_backup_{ts}"
    backup_root.mkdir(parents=True, exist_ok=True)
    backup_paths(rel_paths, backup_root)

    ok = 0
    failed: List[Tuple[str, str]] = []
    for fp in patches:
        old_rp = _normalize_repo_path(fp.old_path)
        new_rp = _normalize_repo_path(fp.new_path)
        if not new_rp:
            continue
        target = repo_root / new_rp

        try:
            if fp.deleted_file:
                if target.exists():
                    target.unlink()
                ok += 1
                continue

            # 如果这是“路径变更（移动/重命名）”的补丁，而新路径还不存在，
            # 则优先用旧路径文件作为基底，再在此基础上应用 hunk。
            seed_path: Optional[Path] = None
            if not fp.new_file and not target.exists() and old_rp and old_rp != new_rp:
                candidate = repo_root / old_rp
                if candidate.exists():
                    seed_path = candidate

            if fp.new_file:
                base_lines = []
            else:
                base_lines = _read_text_lines(seed_path or target)
            out_lines = _apply_hunks(base_lines, fp.hunks, new_rp)
            _write_text_lines(target, out_lines)
            ok += 1
        except Exception as e:
            failed.append((new_rp, str(e)))

    print(f"已解析文件补丁数：{len(patches)}")
    print(f"成功应用：{ok}")
    print(f"失败：{len(failed)}")
    if failed:
        print("失败列表：")
        for p, err in failed[:50]:
            print(p)
            print(err)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
