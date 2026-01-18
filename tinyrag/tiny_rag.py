
import os
import json
import random
from loguru import logger
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from tinyrag import BaseLLM, Qwen2LLM, TinyLLM,qwen3_llm
from tinyrag import Searcher
from tinyrag.searcher.multi_db_searcher import MultiDBSearcher
from tinyrag import SentenceSplitter
from tinyrag.utils import write_list_to_jsonl, resolve_db_dir
from tinyrag.rag import build_context_and_citations, build_hyde_prompt, build_rag_prompt, chunk_doc_item

@dataclass
class RAGConfig:
    # 兼容字段：如直接指定数据库目录，可继续使用 base_dir
    base_dir:str = "data/wiki_db"
    # 新命名规则：db_root_dir/db_name（db_name 默认取 source_path 文件名去后缀）
    db_root_dir: str = "data/db"
    db_name: str = ""
    source_path: str = ""
    llm_model_id:str = "Qwen/Qwen3-8B"
    emb_model_id: str = "models/bge-base-zh-v1.5"
    ranker_model_id:str = "models/bge-reranker-base"
    device:str = "cpu"
    sent_split_model_id:str = "models/nlp_bert_document-segmentation_chinese-base"
    sent_split_use_model:bool = False
    sentence_size:int = 2048
    model_type: str = "qwen3"
    min_chunk_len: int = 20
    multi_db: bool = False
    multi_db_names: List[str] = field(default_factory=list)

    retrieval_strategy: str = "answer_augmented"
    fusion_method: str = "dedup"
    recall_factor: int = 2
    rrf_k: int = 60
    bm25_weight: float = 1.0
    emb_weight: float = 1.0
    hyde_use_as_answer: bool = False

class TinyRAG:
    def __init__(self, config:RAGConfig) -> None:
        print("config: ", config)
        self.config = config

        # 统一数据库命名规则：data/db/<原始文件名>/...
        resolved = resolve_db_dir(
            self.config.db_root_dir,
            source_path=self.config.source_path,
            db_name=self.config.db_name,
        )
        if resolved:
            self.config.base_dir = resolved
        os.makedirs(self.config.base_dir, exist_ok=True)
        logger.info("db dir: {}".format(self.config.base_dir))

        self.searcher: Optional[Searcher] = None
        if not self.config.multi_db:
            self._ensure_single_searcher()
        self._multi_searcher: Optional[MultiDBSearcher] = None

        if self.config.model_type == "qwen2":
            self.llm:BaseLLM = Qwen2LLM(
                model_id_key=config.llm_model_id,
                device=self.config.device
            )
        elif self.config.model_type == "qwen3":
            self.llm:BaseLLM = qwen3_llm(
                model_id_key=config.llm_model_id,
                device=self.config.device
            )
        elif self.config.model_type == "tinyllm":
            self.llm:BaseLLM = TinyLLM(
                model_id_key=config.llm_model_id,
                device=self.config.device
            )
        else:
            raise ValueError("failed init LLM, the model type is [qwen2, qwen3, tinyllm]")

    def _ensure_single_searcher(self) -> None:
        if self.searcher is None:
            self.searcher = Searcher(
                emb_model_id=self.config.emb_model_id,
                ranker_model_id=self.config.ranker_model_id,
                device=self.config.device,
                base_dir=self.config.base_dir,
            )

    def build(self, docs: List[str]):
        """ 注意： 构建数据库需要很长时间
        """
        self._ensure_single_searcher()
        self.sent_split_model = SentenceSplitter(
            use_model=self.config.sent_split_use_model, 
            sentence_size=self.config.sentence_size, 
            model_path=self.config.sent_split_model_id
        )
        logger.info("load sentence splitter model success! ")
        chunk_list: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_to_item = {
                executor.submit(
                    chunk_doc_item,
                    item,
                    self.sent_split_model,
                    min_chunk_len=self.config.min_chunk_len,
                ): item
                for item in docs
            }
            
            for future in tqdm(as_completed(future_to_item), total=len(docs), ascii=True):
                try:
                    chunk_list.extend(future.result())
                except Exception as exc:
                    logger.error(f"Generated an exception: {exc}")

        write_list_to_jsonl(chunk_list, self.config.base_dir + "/split_sentence.jsonl")
        logger.info("split sentence success, all sentence number: {}", len(chunk_list))
        if not chunk_list:
            raise ValueError(
                "构建失败：切分过滤后没有任何文本片段可用于建库。"
                "请检查配置中的 min_chunk_len 是否过大（例如 256 会把英文 PDF 的多数句子过滤掉），"
                "建议先调小到 20/50；或调整 sentence_size/切分策略。"
            )
        logger.info("build database ...... ")
        self.searcher.build_db(chunk_list)
        logger.info("build database success, starting save .... ")
        self.searcher.save_db()
        logger.info("save database success!  ")

    def load(self):
        if self.config.multi_db:
            db_dirs = MultiDBSearcher.discover_db_dirs(
                self.config.db_root_dir,
                names=self.config.multi_db_names if self.config.multi_db_names else None,
            )
            self._multi_searcher = MultiDBSearcher(
                base_dirs=db_dirs,
                emb_model_id=self.config.emb_model_id,
                ranker_model_id=self.config.ranker_model_id,
                device=self.config.device,
            )
            self._multi_searcher.load_all()
            logger.info("multi-db search load database success!")
        else:
            self._ensure_single_searcher()
            self.searcher.load_db()
            logger.info("search load database success!")

    def search(self, query: str, top_n:int = 3) -> str:
        strategy = (self.config.retrieval_strategy or "answer_augmented").lower().strip()
        recall_factor = self.config.recall_factor if self.config.recall_factor and self.config.recall_factor > 0 else 2
        recall_k = max(1, recall_factor * top_n)

        llm_result_txt = ""
        hyde_text = ""
        if strategy == "hyde":
            hyde_prompt = build_hyde_prompt(query)
            hyde_text = (self.llm.generate(hyde_prompt) or "").strip()
            if self.config.hyde_use_as_answer:
                llm_result_txt = hyde_text
            else:
                llm_result_txt = self.llm.generate(query)
        else:
            # 默认沿用原逻辑：先生成初答，再用“query + 初答 + query”做检索
            llm_result_txt = self.llm.generate(query)

        # 构造检索 query（BM25 与向量召回可分开）
        if strategy == "hyde" and hyde_text:
            bm25_query = query
            emb_query_text = hyde_text
            rerank_query = query
            if self.config.multi_db:
                if self._multi_searcher is None:
                    self.load()
                search_content_list = self._multi_searcher.search_advanced(
                    rerank_query=rerank_query,
                    bm25_query=bm25_query,
                    emb_query_text=emb_query_text,
                    top_n=top_n,
                    recall_k=recall_k,
                    fusion_method=self.config.fusion_method or "rrf",
                    rrf_k=self.config.rrf_k,
                    bm25_weight=self.config.bm25_weight,
                    emb_weight=self.config.emb_weight,
                )
            else:
                self._ensure_single_searcher()
                search_content_list = self.searcher.search_advanced(
                    rerank_query=rerank_query,
                    bm25_query=bm25_query,
                    emb_query_text=emb_query_text,
                    top_n=top_n,
                    recall_k=recall_k,
                    fusion_method=self.config.fusion_method or "rrf",
                    rrf_k=self.config.rrf_k,
                    bm25_weight=self.config.bm25_weight,
                    emb_weight=self.config.emb_weight,
                )
        else:
            search_query = (query or "") + (llm_result_txt or "") + (query or "")
            if self.config.multi_db:
                if self._multi_searcher is None:
                    self.load()
                search_content_list = self._multi_searcher.search_advanced(
                    rerank_query=query,
                    bm25_query=search_query,
                    emb_query_text=search_query,
                    top_n=top_n,
                    recall_k=recall_k,
                    fusion_method=self.config.fusion_method or "rrf",
                    rrf_k=self.config.rrf_k,
                    bm25_weight=self.config.bm25_weight,
                    emb_weight=self.config.emb_weight,
                )
            else:
                self._ensure_single_searcher()
                search_content_list = self.searcher.search(query=search_query, top_n=top_n)

        chunks = [item[1] for item in search_content_list]
        context, cite_lines = build_context_and_citations(chunks)
        # 构造 prompt
        prompt_text = build_rag_prompt(context=context, question=query, answer=llm_result_txt)
        logger.info("prompt: {}".format(prompt_text))
        # 生成最终答案
        output = self.llm.generate(prompt_text)
        if cite_lines:
            output = (output or "").rstrip() + "\n\n引用信息如下：\n" + "\n".join(cite_lines)

        return output
        
