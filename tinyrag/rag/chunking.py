from __future__ import annotations

import re
from typing import Any, Dict, List, Union

from ..utils import make_chunk_id, make_doc_id


_RE_LAW_ENUM = re.compile(r"^\s*[（(][一二三四五六七八九十百千0-9]+[)）]\s*")


def _is_law_doc(meta: Dict[str, Any]) -> bool:
    if not isinstance(meta, dict):
        return False
    return bool(meta.get("law") or meta.get("article") or meta.get("book") or meta.get("chapter"))


def _law_index_prefix(meta: Dict[str, Any]) -> str:
    law = str(meta.get("law") or "").strip()
    book = str(meta.get("book") or "").strip()
    chapter = str(meta.get("chapter") or "").strip()
    section = str(meta.get("section") or "").strip() or "未分节"
    article = str(meta.get("article") or "").strip()

    # 常见写法：用户会输入“刑法/宪法/民法典”而不带“中华人民共和国”
    alias = ""
    if law.startswith("中华人民共和国") and len(law) > len("中华人民共和国"):
        alias = law[len("中华人民共和国") :].strip()

    parts = []
    if law:
        parts.append(f"《{law}》")
    if alias and alias != law:
        parts.append(f"（简称：{alias}）")
    for p in (book, chapter, section, article):
        if p:
            parts.append(p)
    return " ".join(parts).strip()


def _merge_law_sentences(
    sents: List[str],
    *,
    max_chars: int,
    min_chars: int = 120,
) -> List[str]:
    """
    针对法条文本做“条文内合并”：
    - 类似“（一）/（二）…”的枚举项不应单独成块，优先合并到同一 chunk
    - 以“：”结尾的引导句与后续枚举项优先同块
    - chunk 尽量达到 min_chars，但不超过 max_chars
    """
    max_chars = max(1, int(max_chars))
    min_chars = max(1, int(min_chars))

    out: List[str] = []
    buf: List[str] = []

    def buf_len() -> int:
        return sum(len(x) for x in buf) + max(0, len(buf) - 1)

    def flush() -> None:
        nonlocal buf
        if not buf:
            return
        text = "\n".join([x.strip() for x in buf if x and x.strip()]).strip()
        if text:
            out.append(text)
        buf = []

    for sent in [s.strip() for s in (sents or []) if s and s.strip()]:
        # 新块起始条件：当前 buffer 已经够长且再加会太长
        cur_len = buf_len()
        if buf and cur_len >= min_chars and (cur_len + 1 + len(sent)) > max_chars:
            flush()

        # 默认追加到当前 buffer
        buf.append(sent)

        # 如果遇到“引导句：”，先别急着 flush，等待至少一个枚举项进来
        if sent.endswith("：") or sent.endswith(":"):
            continue

        # 当 buffer 达到一定长度时，可以考虑 flush，但枚举项不要被拆散得过碎
        if buf_len() >= max_chars:
            flush()

    flush()
    return out


def chunk_doc_item(
    doc_item: Union[str, Dict[str, Any]],
    sent_split_model: Any,
    *,
    min_chunk_len: int,
) -> List[Dict[str, Any]]:
    if isinstance(doc_item, dict):
        text = (doc_item.get("text") or "").strip()
        meta = doc_item.get("meta") or {}
        doc_id = str(doc_item.get("id") or meta.get("doc_id") or "")
        source_path = meta.get("source_path", "")
        page = int(meta.get("page") or 0)
        if not doc_id:
            doc_id = make_doc_id(
                source_path=source_path,
                page=page,
                record_index=int(meta.get("record_index") or 0),
            )
    else:
        text = (doc_item or "").strip()
        meta = {"source_path": ""}
        doc_id = make_doc_id(source_path="", page=0, record_index=0)

    if not text:
        return []

    sent_res = sent_split_model.split_text(text)
    sent_res = [s for s in sent_res if s and s.strip()]

    # 法律条文：先做合并再按 min_chunk_len 过滤，避免“（一）（二）”这类短句被单独成块
    if _is_law_doc(meta):
        sent_res = _merge_law_sentences(sent_res, max_chars=getattr(sent_split_model, "sentence_size", 512))

    sent_res = [s for s in sent_res if s and len(s) >= int(min_chunk_len)]

    out: List[Dict[str, Any]] = []
    law_prefix = _law_index_prefix(meta) if _is_law_doc(meta) else ""
    for idx, sent in enumerate(sent_res):
        out_meta = dict(meta)
        out_meta["chunk_index"] = idx
        chunk_id = make_chunk_id(doc_id=doc_id, chunk_index=idx)
        chunk: Dict[str, Any] = {"id": chunk_id, "text": sent, "meta": out_meta}
        if law_prefix:
            chunk["index_text"] = (law_prefix + "\n" + sent).strip()
        out.append(chunk)
    return out
