from collections import defaultdict
from typing import List, Any, Optional, Dict
import re

from tinyrag.embedding.base_emb import BaseEmbedding

class BaseParser:
    """
    Top class of data parser
    """
    type = None
    def __init__(self, file_path: str, model: BaseEmbedding = None) -> None:
        self.file_path: str = file_path
        self.model: BaseEmbedding = model
        self._metadata: Optional[defaultdict] = None
        self.parse_output: Any = None


    def parse(self) -> List[Dict]:
        raise NotImplementedError()

    # def _to_sentences(self) -> List[Any]:
    #     """
    #     Parse file to sentences
    #     """
    #     raise NotImplementedError()
    
    def _check_format(self) -> bool:
        """
        Check input file format
        """
        raise NotImplementedError()
    
    @property
    def metadata(self) -> defaultdict:
        """
        Parse metadata
        """
        raise NotImplementedError()
    
    def get_embedding(self, obj: Any):
        if self.model is not None:
            return self.model.get_embedding(obj)
        else:
            return None

    @staticmethod
    def split_sentences(text: str, *, sentence_size: int = 256, prefer_zh: bool = True) -> List[str]:
        """
        统一分句入口：
        - 中文（含CJK字符）默认走 SentenceSplitter 的规则切分
        - 非中文优先使用 nltk.sent_tokenize；若缺少资源/不可用则回退到 SentenceSplitter
        """
        text = (text or "").strip()
        if not text:
            return []

        if prefer_zh and re.search(r"[\u4e00-\u9fff]", text):
            from tinyrag.sentence_splitter import SentenceSplitter

            splitter = SentenceSplitter(use_model=False, sentence_size=sentence_size)
            return [s for s in splitter.split_text(text) if s]

        try:
            from nltk.tokenize import sent_tokenize

            return [s for s in sent_tokenize(text) if s]
        except Exception:
            from tinyrag.sentence_splitter import SentenceSplitter

            splitter = SentenceSplitter(use_model=False, sentence_size=sentence_size)
            return [s for s in splitter.split_text(text) if s]
