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
        searchers: List[Searcher],
        llm: Any,
        recall_factor: int = 4,
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        emb_weight: float = 1.0,
    ) -> None:
        self._searchers = searchers
        self._llm = llm
        self._recall_factor = recall_factor
        self._rrf_k = rrf_k
        self._bm25_weight = bm25_weight
        self._emb_weight = emb_weight

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="rag_search",
            description="在当前数据库中进行证据检索（默认策略：HyDE + RRF + rerank），返回带元数据的片段列表。" \
            "目前支持两个数据库：law（法律法规）和 case（司法案例）。" \
            "当你需要从法律法规或司法案例中寻找答案时使用。" \
            "用户询问法律问题时，必须查找law库，而case库可选择作为案例补充使用" \
            "注意，使用case库时,topk不宜过大，推荐为'topk: 3'，以免返回过多无关案例片段影响回答质量。",
        )

    def prompt_usage(self) -> str:
        return (
            "Action Input 必须是 JSON 对象，字段如下：\n"
            "{\n"
            '  "query": "用户问题/检索查询（必填）",\n'
            '  "topk": 5\n'
            '  "db_name": "law" 或 "case" （必填，选择使用的数据库）\n'
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
        db_name = str(kwargs.get("db_name") or "").strip()
        if db_name=="law":
            reranked = self._searchers[0].search_advanced(
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
        elif db_name=="case":
            reranked = self._searchers[1].search_advanced(
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
            description="网页搜索引擎（SerpApi）。当你需要回答时事、事实，或你认为知识库信息不足时使用。" \
            "当用户问题包含'近期'，'最近','最新','现在','当前','当下'等时间词时，考虑使用该工具。",
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
    def format_law_location(meta: Dict[str, Any]) -> str:
        law = str(meta.get("law") or "").strip()
        book = str(meta.get("book") or "").strip() or "未知编"
        chapter = str(meta.get("chapter") or "").strip() or "未知章"
        section = str(meta.get("section") or "").strip() or "未分节"
        article = str(meta.get("article") or "").strip() or "未知条"
        parts = [p for p in [law, book, chapter, section, article] if p]
        return " | ".join(parts)

    def format_source(meta: Dict[str, Any]) -> str:
        source_path = str(meta.get("source_path") or "").strip()
        page = meta.get("page", None)
        if meta.get("law") or meta.get("article") or meta.get("book") or meta.get("chapter"):
            loc = format_law_location(meta)
            if source_path:
                return f"{source_path} | {loc}"
            return loc
        # 案例 PDF：优先展示章节与页范围定位
        if meta.get("pdf_mode") == "case" or meta.get("case_title") or meta.get("case_para_start") or meta.get("case_para_end"):
            parts: List[str] = []
            if source_path:
                parts.append(source_path)
            title = str(meta.get("case_title") or "").strip()
            if title:
                parts.append(title)
            ps = meta.get("page_start")
            pe = meta.get("page_end")
            if ps and pe:
                parts.append(f"第{ps}~{pe}页")
            elif page:
                parts.append(f"第{page}页")
            sections = meta.get("case_sections") or []
            if isinstance(sections, list) and sections:
                uniq: List[str] = []
                seen = set()
                for s in [str(x).strip() for x in sections if str(x).strip()]:
                    if s not in seen:
                        uniq.append(s)
                        seen.add(s)
                if uniq:
                    parts.append("章节=" + ",".join(uniq))
            return " | ".join([p for p in parts if p]).strip() or (source_path or "未知来源")
        if source_path:
            if page:
                return f"{source_path} 第{page}页"
            return source_path
        return "未知来源"

    def expand_case_blocks(meta: Dict[str, Any]) -> str:
        source_path = str(meta.get("source_path") or "").strip()
        if not source_path:
            return ""

        # 缓存：避免同一案例重复解析 PDF
        cache: Dict[str, Dict[str, Any]] = getattr(format_observation_for_prompt, "_case_cache", {})
        if not isinstance(cache, dict):
            cache = {}

        if source_path not in cache:
            try:
                from pathlib import Path

                from tinyrag.ingest.readers.pdf_reader import read_case_pdf_sections

                cache[source_path] = read_case_pdf_sections(Path(source_path))
            except Exception:
                cache[source_path] = {}
            setattr(format_observation_for_prompt, "_case_cache", cache)

        data = cache.get(source_path) or {}
        title = str(data.get("case_title") or meta.get("case_title") or "").strip()
        secs = data.get("sections") or {}
        if not isinstance(secs, dict):
            secs = {}

        def sec_block(name: str) -> str:
            body = str(secs.get(name) or "").strip()
            if not body:
                return ""
            return f"【{name}】\n{body}".strip()

        blocks = [b for b in [sec_block("基本案情"), sec_block("裁判理由"), sec_block("裁判要旨")] if b]
        if not blocks:
            return ""
        head = title.strip()
        if head:
            return (head + "\n" + "\n\n".join(blocks)).strip()
        return "\n\n".join(blocks).strip()

    items = result.get("items") or []
    lines: List[str] = []
    err = result.get("error")
    if err:
        lines.append(f"error={err}")
    seen_case_sources: set[str] = set()
    display_rank = 0
    for item in items:
        meta = item.get("meta") or {}
        if not isinstance(meta, dict):
            meta = {}
        url = meta.get("url", "")

        is_case = bool(meta.get("pdf_mode") == "case" or meta.get("case_title") or meta.get("case_para_start") or meta.get("case_para_end"))
        if is_case:
            source_path = str(meta.get("source_path") or "").strip()
            if source_path and source_path in seen_case_sources:
                continue
            if source_path:
                seen_case_sources.add(source_path)

            expanded = expand_case_blocks(meta)
            if expanded:
                display_rank += 1
                # 案例证据：保留换行，整块给出“基本案情/裁判理由/裁判要旨”
                lines.append(f"[{display_rank}] {expanded}")
                lines.append(f"source={url if url else format_source(meta)}")
                continue

        # 默认：沿用原逻辑（压缩换行并截断）
        display_rank += 1
        text = str(item.get("text") or "").strip().replace("\n", " ")
        if max_chars_per_item and len(text) > max_chars_per_item:
            text = text[:max_chars_per_item] + "..."
        lines.append(f"[{display_rank}] {text}")
        lines.append(f"source={url if url else format_source(meta)}")
    if not items and not err:
        lines.append("（无结果）")
    return "\n".join(lines).strip()
