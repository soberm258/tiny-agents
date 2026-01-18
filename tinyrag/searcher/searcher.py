from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from tqdm import tqdm

from tinyrag.embedding.hf_emb import HFSTEmbedding
from tinyrag.searcher.bm25_recall.bm25_retriever import BM25Retriever
from tinyrag.searcher.emb_recall.emb_retriever import EmbRetriever
from tinyrag.searcher.reranker.reanker_bge_m3 import RerankerBGEM3


BM25RecallItem = Tuple[int, Any, float]
EmbRecallItem = Tuple[int, Any, float]


def _to_text(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("text") or "")
    return str(item or "")


def _item_key(item: Any) -> str:
    if isinstance(item, dict):
        item_id = item.get("id")
        if item_id:
            return f"id:{item_id}"
        meta = item.get("meta") or {}
        if isinstance(meta, dict):
            doc_id = meta.get("doc_id")
            if doc_id:
                return f"doc_id:{doc_id}"
        return f"text:{_to_text(item)}"
    return f"text:{_to_text(item)}"


def rrf_fuse(
    bm25_list: List[BM25RecallItem],
    emb_list: List[EmbRecallItem],
    *,
    top_k: int,
    k: int = 60,
    bm25_weight: float = 1.0,
    emb_weight: float = 1.0,
) -> List[Any]:
    """
    Reciprocal Rank Fusion (RRF)：
    - BM25：按分数从高到低排序
    - 向量：按距离从小到大排序（L2 越小越相似）
    """
    top_k = max(1, int(top_k))
    k = max(1, int(k))

    score_map: Dict[str, float] = {}
    item_map: Dict[str, Any] = {}

    bm25_sorted = sorted(bm25_list, key=lambda x: x[2], reverse=True)
    emb_sorted = sorted(emb_list, key=lambda x: x[2])

    for rank, (_idx, item, _score) in enumerate(bm25_sorted, start=1):
        key = _item_key(item)
        item_map.setdefault(key, item)
        score_map[key] = score_map.get(key, 0.0) + float(bm25_weight) * (1.0 / (k + rank))

    for rank, (_idx, item, _dist) in enumerate(emb_sorted, start=1):
        key = _item_key(item)
        item_map.setdefault(key, item)
        score_map[key] = score_map.get(key, 0.0) + float(emb_weight) * (1.0 / (k + rank))

    fused = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    return [item_map[key] for key, _ in fused[:top_k]]


class Searcher:
    def __init__(
        self,
        *,
        emb_model_id: str,
        ranker_model_id: str,
        device: str = "cpu",
        base_dir: str = "data/db",
    ) -> None:
        self.base_dir = base_dir
        self.device = device

        bm25_dir = os.path.join(self.base_dir, "bm_corpus")
        faiss_dir = os.path.join(self.base_dir, "faiss_idx")

        # 召回
        self.bm25_retriever = BM25Retriever(base_dir=bm25_dir)
        self.emb_model = HFSTEmbedding(path=emb_model_id, device=self.device)
        try:
            index_dim = self.emb_model.st_model.get_sentence_embedding_dimension()
        except Exception:
            index_dim = len(self.emb_model.get_embedding("test_dim"))
        self.emb_retriever = EmbRetriever(index_dim=index_dim, base_dir=faiss_dir)

        # 排序
        self.ranker = RerankerBGEM3(model_id_key=ranker_model_id, device=self.device)

        logger.info("Searcher init build success...")

    def build_db(self, docs: List[Any]) -> None:
        if not docs:
            raise ValueError("构建失败：docs 为空，无法构建 BM25/向量索引。")

        self.bm25_retriever.build(docs)
        logger.info("bm25 retriever build success...")

        device_lower = str(self.device).lower()
        default_bs = "96" if "cuda" in device_lower else "16"
        batch_size = int(os.getenv("TINYRAG_EMB_BATCH_SIZE", default_bs))
        batch_size = max(1, batch_size)

        for start in tqdm(range(0, len(docs), batch_size), desc="emb build ", ascii=True):
            batch_docs = docs[start : start + batch_size]
            batch_texts = [_to_text(x) for x in batch_docs]
            batch_embs = self.emb_model.get_embeddings(batch_texts, batch_size=batch_size)
            self.emb_retriever.batch_insert(batch_embs, batch_docs)

        logger.info("emb retriever build success...")

    def save_db(self) -> None:
        self.bm25_retriever.save_bm25_data()
        logger.info("bm25 retriever save success...")
        self.emb_retriever.save()
        logger.info("emb retriever save success...")

    def load_db(self) -> None:
        self.bm25_retriever.load_bm25_data()
        logger.info("bm25 retriever load success...")
        self.emb_retriever.load()
        logger.info("emb retriever load success...")

    def search_advanced(
        self,
        *,
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

        bm25_recall_list = self.bm25_retriever.search(bm25_query, recall_k)
        logger.info("bm25 recall text num: {}", len(bm25_recall_list))

        query_emb = self.emb_model.get_embedding(emb_query_text)
        emb_recall_list = self.emb_retriever.search(query_emb, recall_k)
        logger.info("emb recall text num: {}", len(emb_recall_list))

        fusion_method = (fusion_method or "rrf").lower().strip()
        if fusion_method == "rrf":
            candidate_items = rrf_fuse(
                bm25_recall_list,
                emb_recall_list,
                top_k=recall_k,
                k=rrf_k,
                bm25_weight=bm25_weight,
                emb_weight=emb_weight,
            )
        else:
            seen = set()
            candidate_items = []
            for _idx, item, _score in sorted(bm25_recall_list, key=lambda x: x[2], reverse=True):
                key = _item_key(item)
                if key not in seen:
                    candidate_items.append(item)
                    seen.add(key)
            for _idx, item, _dist in sorted(emb_recall_list, key=lambda x: x[2]):
                key = _item_key(item)
                if key not in seen:
                    candidate_items.append(item)
                    seen.add(key)
            candidate_items = candidate_items[:recall_k]

        logger.info("fusion candidate text num: {}", len(candidate_items))
        return self.ranker.rank(rerank_query, candidate_items, top_n)

    def search(self, query: str, top_n: int = 3) -> List[Tuple[float, Any]]:
        return self.search_advanced(
            rerank_query=query,
            bm25_query=query,
            emb_query_text=query,
            top_n=top_n,
            recall_k=2 * max(1, int(top_n)),
            fusion_method="dedup",
        )

