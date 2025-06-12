from abc import ABC, abstractmethod
from langchain.schema import Document
from typing import List, Tuple


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, docs: List[Document], top_n: int) -> List[Document]:
        """query와 docs 리스트를 받아 relevance 기준 상위 top_n 문서를 반환"""
        pass
