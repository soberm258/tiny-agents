from __future__ import annotations

from typing import Any, List, Tuple

from ...logging_utils import logger

from .types import BM25RecallItem, EmbRecallItem


class SingleDBRecallProvider:
    def __init__(self, *, bm25_retriever: Any, emb_model: Any, emb_retriever: Any) -> None:
        self._bm25 = bm25_retriever
        self._emb_model = emb_model
        self._emb = emb_retriever

    def recall(self, *, bm25_query: str, emb_query_text: str, recall_k: int) -> Tuple[List[BM25RecallItem], List[EmbRecallItem]]:
        recall_k = max(1, int(recall_k))

        bm25_recall_list = self._bm25.search(bm25_query, recall_k)
        logger.info("bm25 recall text num: {}", len(bm25_recall_list))

        query_emb = self._emb_model.get_embedding(emb_query_text)
        emb_recall_list = self._emb.search(query_emb, recall_k)
        logger.info("emb recall text num: {}", len(emb_recall_list))

        return bm25_recall_list, emb_recall_list
