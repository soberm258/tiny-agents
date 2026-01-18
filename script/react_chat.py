import sys

sys.path.append(".")

import argparse
import json
import os
import time

from tinyrag.llm.qwen3_llm import qwen3_llm
from tinyrag.searcher.searcher import Searcher
from tinyrag.utils import resolve_db_dir

from agent.react_agent import ReActAgent
from agent.tool_executor import ToolExecutor
from agent.tools import RAGSearchTool, SearchOnlineTool


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="TinyRAG ReAct Agent Demo（单库，默认 HyDE+RRF+rerank）")
    parser.add_argument("--config", type=str, default="config/qwen3_config.json")
    parser.add_argument("--db-name", type=str, required=True, help="data/db 下的库名（目录名）")
    parser.add_argument("--db-root-dir", type=str, default="", help="默认读取 config 中的 db_root_dir")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--llm-timeout-sec", type=int, default=180)
    parser.add_argument("--question", type=str, default="", help="非交互模式：直接回答一次并退出")
    parser.add_argument(
        "--show-steps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否输出 ReAct 每一步（默认开启）",
    )
    args = parser.parse_args()

    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    cfg = read_json(args.config)
    db_root_dir = args.db_root_dir.strip() or str(cfg.get("db_root_dir") or "data/db")
    base_dir = resolve_db_dir(db_root_dir, db_name=args.db_name)
    if not base_dir or not os.path.isdir(base_dir):
        raise FileNotFoundError(f"数据库目录不存在：{base_dir}")

    print(f"数据库目录：{base_dir}", flush=True)
    faiss_dir = os.path.join(base_dir, "faiss_idx")
    if os.path.isdir(faiss_dir):
        try:
            idx_dir = os.path.join(faiss_dir, "index_768")
            inv = os.path.join(idx_dir, "invert_index.faiss")
            fwd = os.path.join(idx_dir, "forward_index.txt")
            if os.path.isfile(inv):
                print(f"向量索引文件：{inv}（约 {os.path.getsize(inv)/1024/1024:.1f} MB）", flush=True)
            if os.path.isfile(fwd):
                print(f"向量前排索引：{fwd}（约 {os.path.getsize(fwd)/1024/1024:.1f} MB）", flush=True)
        except Exception:
            pass

    device = str(cfg.get("device") or "cpu")
    searcher = Searcher(
        emb_model_id=str(cfg["emb_model_id"]),
        ranker_model_id=str(cfg["ranker_model_id"]),
        device=device,
        base_dir=base_dir,
    )

    print("开始加载数据库（大库可能需要几十秒到数分钟）...", flush=True)
    t0 = time.time()
    searcher.load_db()
    print(f"数据库加载完成，耗时 {time.time()-t0:.1f}s", flush=True)

    llm = qwen3_llm(model_id_key=str(cfg.get("llm_model_id") or "Qwen/Qwen3-8B"), device=device)

    rag_tool = RAGSearchTool(
        searcher=searcher,
        llm=llm,
        recall_factor=int(cfg.get("recall_factor") or 4),
        rrf_k=int(cfg.get("rrf_k") or 60),
        bm25_weight=float(cfg.get("bm25_weight") or 1.0),
        emb_weight=float(cfg.get("emb_weight") or 1.0),
    )
    executor = ToolExecutor()
    executor.register(rag_tool)
    executor.register(SearchOnlineTool())

    agent = ReActAgent(
        llm=llm,
        tool_executor=executor,
        max_steps=args.max_steps,
        default_topk=args.topk,
        max_tool_calls=2,
        llm_timeout_sec=args.llm_timeout_sec,
    )

    print("ReAct agent 就绪。", flush=True)
    if args.question.strip():
        answer, _history = agent.run(args.question.strip(), show_steps=args.show_steps)
        if not args.show_steps:
            print(answer)
        return

    print("输入 exit/quit 退出。", flush=True)
    while True:
        try:
            q = input("\n用户> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit", "q"):
            break

        answer, _history = agent.run(q, show_steps=args.show_steps)
        if not args.show_steps:
            print("\n助手>\n" + answer)


if __name__ == "__main__":
    main()

