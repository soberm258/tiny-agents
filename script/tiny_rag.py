import sys
sys.path.append(".")

import os
import json
import random
import argparse
from loguru import logger
from tqdm import tqdm

from tinyrag import RAGConfig, TinyRAG
from tinyrag.ingest import load_docs_for_build

from tinyrag.utils import read_json_to_list

def build_db(config_path, data_path):
    # config_path = "config/build_config.json"
    config = read_json_to_list(config_path)
    config["source_path"] = data_path
    rag_config = RAGConfig(**config)
    tiny_rag = TinyRAG(config=rag_config)

    docs = load_docs_for_build(
        data_path,
        json_text_key=config.get("json_text_key", "completion"),
        recursive=True,
    )
    logger.info("load docs success, doc num: {}".format(len(docs)))
    tiny_rag.build(docs)

def query_search(config_path, data_path, multi_db: bool = False):
    # config_path = "config/tiny_llm_config.json"
    config = read_json_to_list(config_path)
    # 新命名规则：data/db/<原始文件名>/...
    config["source_path"] = data_path
    if multi_db:
        config["multi_db"] = True
    rag_config = RAGConfig(**config)
    tiny_rag = TinyRAG(config=rag_config)
    logger.info("tiny rag init success!")
    tiny_rag.load()
    query = "请介绍一下南京"
    output = tiny_rag.search(query, top_n=6)
    print("output: ", output)

def main():
    parser = argparse.ArgumentParser(description='Tiny RAG Argument Parser')
    parser.add_argument("-c", '--config', type=str, default="config/qwen2_config.json", help='Tiny RAG config')
    parser.add_argument("-t", '--type', type=str, default="search", help='Tiny RAG Type [build, search]')
    parser.add_argument('-p', "--path",  type=str, default="data/raw_data/wikipedia-cn-20230720-filtered.json", help='Tiny RAG data path')
    parser.add_argument("--multi-db", action="store_true", help="分库检索：在 data/db 下遍历多个数据库实例目录进行检索")

    args = parser.parse_args()

    if args.type == "build":
        build_db(args.config, args.path)
    elif args.type == "search":
        query_search(args.config, args.path, multi_db=args.multi_db)

if __name__ == "__main__":
    main()
