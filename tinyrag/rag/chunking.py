from __future__ import annotations

from typing import Any, Dict, List, Union

from ..utils import make_chunk_id, make_doc_id


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
    sent_res = [s for s in sent_res if s and len(s) >= int(min_chunk_len)]

    out: List[Dict[str, Any]] = []
    for idx, sent in enumerate(sent_res):
        out_meta = dict(meta)
        out_meta["chunk_index"] = idx
        chunk_id = make_chunk_id(doc_id=doc_id, chunk_index=idx)
        out.append({"id": chunk_id, "text": sent, "meta": out_meta})
    return out
