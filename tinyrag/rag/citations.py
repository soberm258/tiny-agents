from __future__ import annotations

from typing import Any, Dict, List, Tuple


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
                if page:
                    cite_lines.append(f"[{i}] {src} 第{page}页")
                else:
                    cite_lines.append(f"[{i}] {src}")
            else:
                cite_lines.append(f"[{i}] 未知来源")
        else:
            context_lines.append(f"[{i}] {str(chunk)}")
            cite_lines.append(f"[{i}] 未知来源")

    return "\n".join(context_lines), cite_lines

