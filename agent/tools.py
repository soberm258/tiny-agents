from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from tinyrag.searcher.searcher import Searcher

from .prompts import build_hyde_prompt
from .tool_base import BaseTool, ToolSpec


class RAGSearchTool(BaseTool):
    def __init__(
        self,
        *,
        searcher: "Searcher",
        llm: Any,
        recall_factor: int = 4,
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        emb_weight: float = 1.0,
    ) -> None:
        self._searcher = searcher
        self._llm = llm
        self._recall_factor = recall_factor
        self._rrf_k = rrf_k
        self._bm25_weight = bm25_weight
        self._emb_weight = emb_weight

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="rag_search",
            description="在当前数据库中进行证据检索（默认策略：HyDE + RRF + rerank），返回带元数据的片段列表。",
        )

    def prompt_usage(self) -> str:
        return (
            "Action Input 必须是 JSON 对象，字段如下：\n"
            "{\n"
            '  "query": "用户问题/检索查询（必填）",\n'
            '  "topk": 5\n'
            "}\n"
        )

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        query = str(kwargs.get("query") or "").strip()
        if not query:
            raise ValueError("rag_search.query 不能为空")
        topk = max(1, int(kwargs.get("topk") or 5))

        recall_factor = self._recall_factor if self._recall_factor and self._recall_factor > 0 else 4
        recall_k = max(1, topk * recall_factor)

        hyde_prompt = build_hyde_prompt(query)
        hyde_text = (self._llm.generate(hyde_prompt) or "").strip()
        if (not hyde_text) or ("生成失败" in hyde_text) or ("API调用失败" in hyde_text):
            hyde_text = query

        reranked = self._searcher.search_advanced(
            rerank_query=query,
            bm25_query=query,
            emb_query_text=hyde_text,
            top_n=topk,
            recall_k=recall_k,
            fusion_method="rrf",
            rrf_k=self._rrf_k,
            bm25_weight=self._bm25_weight,
            emb_weight=self._emb_weight,
        )

        items: List[Dict[str, Any]] = []
        for i, (score, item) in enumerate(reranked, start=1):
            if isinstance(item, dict):
                text = str(item.get("text") or "")
                meta = item.get("meta") or {}
                chunk_id = item.get("id") or ""
            else:
                text = str(item or "")
                meta = {}
                chunk_id = ""
            items.append(
                {
                    "rank": i,
                    "score": float(score),
                    "id": str(chunk_id),
                    "text": text,
                    "meta": meta,
                }
            )

        return {
            "query": query,
            "hyde_text": hyde_text[:400],
            "topk": topk,
            "items": items,
        }


def _read_env_key_from_dotenv(*, key: str, dotenv_path: str = ".env") -> str:
    try:
        if not os.path.isfile(dotenv_path):
            return ""
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == key:
                    return v.strip().strip("'").strip('"')
    except Exception:
        return ""
    return ""


class SearchOnlineTool(BaseTool):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="search_online",
            description="网页搜索引擎（SerpApi）。当你需要回答时事、事实，或你认为知识库信息不足时使用。",
        )

    def prompt_usage(self) -> str:
        return (
            "Action Input 必须是 JSON 对象，字段如下：\n"
            "{\n"
            '  "query": "搜索关键词（必填）",\n'
            '  "topk": 5\n'
            "}\n"
        )

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        query = str(kwargs.get("query") or "").strip()
        if not query:
            raise ValueError("search_online.query 不能为空")
        topk = max(1, int(kwargs.get("topk") or 5))

        api_key = (
            os.getenv("SERPAPI_API_KEY")
            or os.getenv("SERPAPI_KEY")
            or _read_env_key_from_dotenv(key="SERPAPI_API_KEY")
            or _read_env_key_from_dotenv(key="SERPAPI_KEY")
        )
        if not api_key:
            return {"query": query, "topk": topk, "items": [], "error": "SERPAPI_API_KEY 或 SERPAPI_KEY 未配置"}

        try:
            from serpapi import Client  # type: ignore
        except Exception as e:
            return {"query": query, "topk": topk, "items": [], "error": f"serpapi 依赖不可用：{e}"}

        params = {"engine": "google", "q": query, "api_key": api_key, "num": topk}
        try:
            client = Client(api_key=api_key)  # type: ignore
            result = client.search(params)  # type: ignore
            if hasattr(result, "as_dict"):
                data = result.as_dict() or {}
            elif isinstance(result, dict):
                data = result
            else:
                data = {}
        except Exception as e:
            return {"query": query, "topk": topk, "items": [], "error": f"SerpApi 调用失败：{e}"}

        if isinstance(data, dict):
            if data.get("error"):
                return {"query": query, "topk": topk, "items": [], "error": str(data.get("error"))}
            meta = data.get("search_metadata") or {}
            if isinstance(meta, dict) and str(meta.get("status") or "").lower() == "error":
                return {"query": query, "topk": topk, "items": [], "error": str(meta.get("error") or "SerpApi 返回错误")}

        organic = data.get("organic_results") or []
        items: List[Dict[str, Any]] = []
        for i, r in enumerate(organic[:topk], start=1):
            title = str(r.get("title") or "")
            link = str(r.get("link") or "")
            snippet = str(r.get("snippet") or "")
            text = " | ".join([x for x in [title, snippet] if x]).strip()
            items.append(
                {
                    "rank": i,
                    "score": 0.0,
                    "id": "",
                    "text": text,
                    "meta": {"url": link, "source_path": "online"},
                }
            )

        if not items:
            return {"query": query, "topk": topk, "items": [], "error": "未获取到搜索结果（可能是 key 无效/额度不足/网络问题）"}
        return {"query": query, "topk": topk, "items": items}


def format_observation_for_prompt(result: Dict[str, Any], *, max_chars_per_item: int = 500) -> str:
    items = result.get("items") or []
    lines: List[str] = []
    err = result.get("error")
    if err:
        lines.append(f"error={err}")
    for item in items:
        rank = item.get("rank")
        text = str(item.get("text") or "").strip().replace("\n", " ")
        if max_chars_per_item and len(text) > max_chars_per_item:
            text = text[:max_chars_per_item] + "..."
        meta = item.get("meta") or {}
        source_path = meta.get("source_path", "")
        page = meta.get("page", None)
        url = meta.get("url", "")
        lines.append(f"[{rank}] {text}")
        if url:
            lines.append(f"url={url}")
        else:
            lines.append(f"source_path={source_path} page={page}")
    if not items and not err:
        lines.append("（无结果）")
    return "\n".join(lines).strip()
