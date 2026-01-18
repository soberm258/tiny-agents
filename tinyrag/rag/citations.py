from __future__ import annotations

from typing import Any, Dict, List, Tuple


def _format_law_location(meta: Dict[str, Any]) -> str:
    law = str(meta.get("law") or "").strip()
    book = str(meta.get("book") or "").strip() or "未知编"
    chapter = str(meta.get("chapter") or "").strip() or "未知章"
    section = str(meta.get("section") or "").strip() or "未分节"
    article = str(meta.get("article") or "").strip() or "未知条"

    parts = [p for p in [law, book, chapter, section, article] if p]
    return " | ".join(parts)


def build_context_and_citations(chunks: List[Any]) -> Tuple[str, List[str]]:
    context_lines: List[str] = []
    cite_lines: List[str] = []

    for i, chunk in enumerate(chunks, start=1):
        if isinstance(chunk, dict):
            text = (chunk.get("text") or "").strip()
            meta = chunk.get("meta") or {}
            src = meta.get("source_path") or ""
            page = meta.get("page") or ""
            context_lines.append(f"[{i}] {text}")
            if src:
                # 法律类（例如民法典）优先按 编/章/节/条 定位；否则沿用 page/file 引用
                if isinstance(meta, dict) and (meta.get("article") or meta.get("law") or meta.get("book")):
                    cite_lines.append(f"[{i}] {src} | {_format_law_location(meta)}")
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
