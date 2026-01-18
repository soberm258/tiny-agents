"""
RAG 相关的可复用组件（尽量保持无副作用、可测试）。

TinyRAG 作为编排器会依赖这里的模块来完成：
1) 提示词构造
2) 文档切分为 chunk
3) 检索结果的上下文与引用格式化
"""

from .prompts import build_hyde_prompt, build_rag_prompt
from .chunking import chunk_doc_item
from .citations import build_context_and_citations

__all__ = [
    "build_hyde_prompt",
    "build_rag_prompt",
    "chunk_doc_item",
    "build_context_and_citations",
]

