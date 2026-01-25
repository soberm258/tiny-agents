from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple


def _format_law_location(meta: Dict[str, Any]) -> str:
    law = str(meta.get("law") or "").strip()
    book = str(meta.get("book") or "").strip() or "未知编"
    chapter = str(meta.get("chapter") or "").strip() or "未知章"
    section = str(meta.get("section") or "").strip() or "未分节"
    article = str(meta.get("article") or "").strip() or "未知条"

    parts = [p for p in [law, book, chapter, section, article] if p]
    return " | ".join(parts)


def _is_case_chunk(meta: Dict[str, Any]) -> bool:
    if not isinstance(meta, dict):
        return False
    return bool(
        meta.get("pdf_mode") == "case"
        or meta.get("case_title")
        or meta.get("case_para_start")
        or meta.get("case_para_end")
    )


def _format_case_source(meta: Dict[str, Any]) -> str:
    src = str(meta.get("source_path") or "").strip()
    title = str(meta.get("case_title") or "").strip()
    ps = meta.get("page_start")
    pe = meta.get("page_end")

    parts = [p for p in [src, title] if p]
    if ps and pe:
        parts.append(f"第{ps}~{pe}页")
    sections = meta.get("case_sections") or []
    if isinstance(sections, list) and sections:
        uniq: List[str] = []
        seen = set()
        for s in [str(x).strip() for x in sections if str(x).strip()]:
            if s not in seen:
                uniq.append(s)
                seen.add(s)
        if uniq:
            parts.append("章节=" + ",".join(uniq))
    return " | ".join(parts).strip() or (src or "未知来源")


def _expand_case_blocks(meta: Dict[str, Any], *, max_chars: int = 6000) -> str:
    src = str(meta.get("source_path") or "").strip()
    if not src:
        return ""
    try:
        from tinyrag.ingest.readers.pdf_reader import read_case_pdf_sections
    except Exception:
        return ""
    try:
        data = read_case_pdf_sections(Path(src))
    except Exception:
        return ""

    title = str(data.get("case_title") or meta.get("case_title") or "").strip()
    secs = data.get("sections") or {}
    if not isinstance(secs, dict):
        secs = {}

    def sec_block(name: str) -> str:
        body = str(secs.get(name) or "").strip()
        if not body:
            return ""
        return f"【{name}】\n{body}".strip()

    blocks = [b for b in [sec_block("基本案情"), sec_block("裁判理由"), sec_block("裁判要旨")] if b]
    if not blocks:
        return ""
    text = ((title + "\n") if title else "") + "\n\n".join(blocks)
    text = text.strip()
    if max_chars and len(text) > max_chars:
        text = text[:max_chars].rstrip() + "…（已截断）"
    return text


def build_context_and_citations(chunks: List[Any]) -> Tuple[str, List[str]]:
    context_lines: List[str] = []
    cite_lines: List[str] = []

    for i, chunk in enumerate(chunks, start=1):
        if isinstance(chunk, dict):
            meta = chunk.get("meta") or {}
            src = meta.get("source_path") or ""
            page = meta.get("page") or ""
            text = (chunk.get("text") or "").strip()

            # 案例 PDF：优先把“基本案情/裁判理由/裁判要旨”整块提供给大模型
            if isinstance(meta, dict) and _is_case_chunk(meta):
                expanded = _expand_case_blocks(meta)
                context_lines.append(f"[{i}] {expanded or text}")
            else:
                context_lines.append(f"[{i}] {text}")
            if src:
                # 法律类（例如民法典）优先按 编/章/节/条 定位；否则沿用 page/file 引用
                if isinstance(meta, dict) and (meta.get("article") or meta.get("law") or meta.get("book")):
                    cite_lines.append(f"[{i}] {src} | {_format_law_location(meta)}")
                elif isinstance(meta, dict) and _is_case_chunk(meta):
                    cite_lines.append(f"[{i}] {_format_case_source(meta)}")
                elif page:
                    cite_lines.append(f"[{i}] {src} 第{page}页")
                else:
                    cite_lines.append(f"[{i}] {src}")
            else:
                cite_lines.append(f"[{i}] 未知来源")
        else:
            context_lines.append(f"[{i}] {str(chunk)}")
            cite_lines.append(f"[{i}] 未知来源")

    return "\n".join(context_lines), cite_lines
