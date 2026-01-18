from __future__ import annotations

from typing import Any, List, Optional, Tuple

from ...logging_utils import logger

from .types import BM25RecallItem, EmbRecallItem


class MultiDBRecallProvider:
    def __init__(self, *, bm25_list: List[Any], emb_list: List[Any], emb_model: Any) -> None:
        self._bm25_list = bm25_list
        self._emb_list = emb_list
        self._emb_model = emb_model

    def recall(self, *, bm25_query: str, emb_query_text: str, recall_k: int) -> Tuple[List[BM25RecallItem], List[EmbRecallItem]]:
        recall_k = max(1, int(recall_k))

        db_num = max(1, len(self._bm25_list))
        per_db_k = max(1, (recall_k + db_num - 1) // db_num)

        bm25_all: List[BM25RecallItem] = []
        for bm25 in self._bm25_list:
            try:
                bm25_all.extend(bm25.search(bm25_query, per_db_k))
            except Exception as e:
                logger.error("BM25召回失败：{}", str(e))

        query_emb = self._emb_model.get_embedding(emb_query_text)
        emb_all: List[EmbRecallItem] = []
        for emb in self._emb_list:
            try:
                emb_all.extend(emb.search(query_emb, per_db_k))
            except Exception as e:
                logger.error("向量召回失败：{}", str(e))

        return bm25_all, emb_all
