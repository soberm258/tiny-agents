from __future__ import annotations

from typing import Any, List, Protocol, Tuple


BM25RecallItem = Tuple[int, Any, float]
EmbRecallItem = Tuple[int, Any, float]


class RecallProvider(Protocol):
    def recall(self, *, bm25_query: str, emb_query_text: str, recall_k: int) -> Tuple[List[BM25RecallItem], List[EmbRecallItem]]: ...

