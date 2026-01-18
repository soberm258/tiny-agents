"""
tinyrag 顶层包尽量保持“轻导入”。

原因：
1) embedding/llm/reranker 等模块依赖 torch/transformers 等重依赖；
2) ingest/结构化解析等轻量功能应当在无 torch 环境下也可用。

因此这里采用延迟导入（PEP 562 __getattr__），在真正访问相关符号时才加载对应子模块。
"""

from __future__ import annotations

from typing import Any


__all__ = [
    # 轻量：可在无 torch 环境下使用
    "SentenceSplitter",
    # RAG 编排器（依赖 Searcher/LLM，访问时才会触发重依赖导入）
    "RAGConfig",
    "TinyRAG",
    # Searcher（依赖 embedding/reranker）
    "Searcher",
    "MultiDBSearcher",
    # LLM 抽象与实现（依赖 transformers/openai 等）
    "BaseLLM",
    "Qwen2LLM",
    "qwen3_llm",
    "TinyLLM",
    # Embedding
    "BaseEmbedding",
    "HFSTEmbedding",
    "ImgEmbedding",
    "OpenAIEmbedding",
    "ZhipuEmbedding",
    # Parser
    "BaseParser",
    "PDFParser",
    "WordParser",
    "PPTXParser",
    "MDParser",
    "TXTParser",
    "ImgParser",
    "parser_file",
]


def __getattr__(name: str) -> Any:
    if name == "SentenceSplitter":
        from .sentence_splitter import SentenceSplitter

        return SentenceSplitter

    if name in ("RAGConfig", "TinyRAG"):
        from .tiny_rag import RAGConfig, TinyRAG

        return {"RAGConfig": RAGConfig, "TinyRAG": TinyRAG}[name]

    if name == "Searcher":
        from .searcher.searcher import Searcher

        return Searcher
    if name == "MultiDBSearcher":
        from .searcher.multi_db_searcher import MultiDBSearcher

        return MultiDBSearcher

    if name == "BaseLLM":
        from .llm.base_llm import BaseLLM

        return BaseLLM
    if name == "Qwen2LLM":
        from .llm.qwen2_llm import Qwen2LLM

        return Qwen2LLM
    if name == "qwen3_llm":
        from .llm.qwen3_llm import qwen3_llm

        return qwen3_llm
    if name == "TinyLLM":
        from .llm.tiny_llm import TinyLLM

        return TinyLLM

    if name in ("BaseEmbedding", "HFSTEmbedding", "ImgEmbedding", "OpenAIEmbedding", "ZhipuEmbedding"):
        from .embedding.base_emb import BaseEmbedding
        from .embedding.hf_emb import HFSTEmbedding
        from .embedding.img_emb import ImgEmbedding
        from .embedding.openai_emb import OpenAIEmbedding
        from .embedding.zhipu_emb import ZhipuEmbedding

        return {
            "BaseEmbedding": BaseEmbedding,
            "HFSTEmbedding": HFSTEmbedding,
            "ImgEmbedding": ImgEmbedding,
            "OpenAIEmbedding": OpenAIEmbedding,
            "ZhipuEmbedding": ZhipuEmbedding,
        }[name]

    if name in ("BaseParser", "PDFParser", "WordParser", "PPTXParser", "MDParser", "TXTParser", "ImgParser", "parser_file"):
        from .parser.base_parser import BaseParser
        from .parser.pdf_parser import PDFParser
        from .parser.doc_parser import WordParser
        from .parser.ppt_parser import PPTXParser
        from .parser.md_parser import MDParser
        from .parser.txt_parser import TXTParser
        from .parser.img_parser import ImgParser
        from .parser import parser_file

        return {
            "BaseParser": BaseParser,
            "PDFParser": PDFParser,
            "WordParser": WordParser,
            "PPTXParser": PPTXParser,
            "MDParser": MDParser,
            "TXTParser": TXTParser,
            "ImgParser": ImgParser,
            "parser_file": parser_file,
        }[name]

    raise AttributeError(name)
