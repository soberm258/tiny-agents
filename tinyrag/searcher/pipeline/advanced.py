from __future__ import annotations

from typing import Any, List, Optional, Tuple

from ...logging_utils import logger

from ..fusion import dedup_fuse, rrf_fuse
from ..recall.types import BM25RecallItem, EmbRecallItem, RecallProvider


def _fuse_candidates(
    bm25_list: List[BM25RecallItem],
    emb_list: List[EmbRecallItem],
    *,
    recall_k: int,
    fusion_method: str,
    rrf_k: int,
    bm25_weight: float,
    emb_weight: float,
) -> List[Any]:
    fusion_method = (fusion_method or "rrf").lower().strip()
    if fusion_method == "rrf":
        return rrf_fuse(
            bm25_list,
            emb_list,
            top_k=recall_k,
            k=rrf_k,
            bm25_weight=bm25_weight,
            emb_weight=emb_weight,
        )
    return dedup_fuse(bm25_list, emb_list, top_k=recall_k)


def run_search_advanced(
    *,
    recall_provider: RecallProvider,
    ranker: Any,
    rerank_query: str,
    bm25_query: str,
    emb_query_text: str,
    top_n: int = 3,
    recall_k: Optional[int] = None,
    fusion_method: str = "rrf",
    rrf_k: int = 60,
    bm25_weight: float = 1.0,
    emb_weight: float = 1.0,
) -> List[Tuple[float, Any]]:
    top_n = max(1, int(top_n))
    recall_k = int(recall_k) if recall_k is not None else 2 * top_n
    recall_k = max(top_n, recall_k)

    bm25_list, emb_list = recall_provider.recall(bm25_query=bm25_query, emb_query_text=emb_query_text, recall_k=recall_k)

    candidates = _fuse_candidates(
        bm25_list,
        emb_list,
        recall_k=recall_k,
        fusion_method=fusion_method,
        rrf_k=rrf_k,
        bm25_weight=bm25_weight,
        emb_weight=emb_weight,
    )
    logger.info("fusion candidate text num: {}", len(candidates))
    return ranker.rank(rerank_query, candidates, top_n)
