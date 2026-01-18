import os
from typing import Dict, List, Optional, Tuple

from tinyrag.logging_utils import logger

from tinyrag.embedding.hf_emb import HFSTEmbedding
from tinyrag.searcher.bm25_recall.bm25_retriever import BM25Retriever
from tinyrag.searcher.emb_recall.emb_retriever import EmbRetriever
from tinyrag.searcher.reranker.reanker_bge_m3 import RerankerBGEM3
from tinyrag.searcher.pipeline import run_search_advanced
from tinyrag.searcher.recall import MultiDBRecallProvider


class MultiDBSearcher:
    """
    分库检索：从多个数据库实例目录召回候选，融合后再 rerank。
    目录结构假设为：
      <base_dir>/bm_corpus
      <base_dir>/faiss_idx
    """

    def __init__(
        self,
        *,
        base_dirs: List[str],
        emb_model_id: str,
        ranker_model_id: str,
        device: str = "cpu",
    ) -> None:
        self.base_dirs = [d for d in base_dirs if d]
        self.device = device

        self.emb_model = HFSTEmbedding(path=emb_model_id, device=device)
        try:
            self.index_dim = self.emb_model.st_model.get_sentence_embedding_dimension()
        except Exception:
            self.index_dim = len(self.emb_model.get_embedding("test_dim"))

        self.ranker = RerankerBGEM3(model_id_key=ranker_model_id, device=device)

        self._bm25_list: List[Tuple[str, BM25Retriever]] = []
        self._emb_list: List[Tuple[str, EmbRetriever]] = []

        for base_dir in self.base_dirs:
            bm25_dir = os.path.join(base_dir, "bm_corpus")
            faiss_dir = os.path.join(base_dir, "faiss_idx")
            if not (os.path.isdir(bm25_dir) and os.path.isdir(faiss_dir)):
                logger.warning("跳过不完整数据库目录：{}", base_dir)
                continue
            self._bm25_list.append((base_dir, BM25Retriever(base_dir=bm25_dir)))
            self._emb_list.append((base_dir, EmbRetriever(index_dim=self.index_dim, base_dir=faiss_dir)))

        logger.info("MultiDBSearcher init success, db num: {}", len(self._bm25_list))

    @staticmethod
    def discover_db_dirs(db_root_dir: str, *, names: Optional[List[str]] = None) -> List[str]:
        if not db_root_dir or not os.path.isdir(db_root_dir):
            return []
        if names:
            return [os.path.join(db_root_dir, n) for n in names]
        dirs: List[str] = []
        for name in os.listdir(db_root_dir):
            p = os.path.join(db_root_dir, name)
            if os.path.isdir(p):
                dirs.append(p)
        return sorted(dirs)

    def load_all(self) -> None:
        ok = 0
        for base_dir, bm25 in self._bm25_list:
            try:
                bm25.load_bm25_data()
                ok += 1
            except Exception as e:
                logger.error("BM25加载失败：{}，错误：{}", base_dir, str(e))

        for base_dir, emb in self._emb_list:
            try:
                emb.load()
            except Exception as e:
                logger.error("向量索引加载失败：{}，错误：{}", base_dir, str(e))

        logger.info("MultiDBSearcher load complete, bm25 ok: {}", ok)

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
    ) -> list:
        recall_provider = MultiDBRecallProvider(
            bm25_list=[bm25 for _base, bm25 in self._bm25_list],
            emb_list=[emb for _base, emb in self._emb_list],
            emb_model=self.emb_model,
        )
        return run_search_advanced(
            recall_provider=recall_provider,
            ranker=self.ranker,
            rerank_query=rerank_query,
            bm25_query=bm25_query,
            emb_query_text=emb_query_text,
            top_n=top_n,
            recall_k=recall_k,
            fusion_method=fusion_method,
            rrf_k=rrf_k,
            bm25_weight=bm25_weight,
            emb_weight=emb_weight,
        )
