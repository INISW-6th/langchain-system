from abc import ABC, abstractmethod
from langchain.schema import Document
from typing import List, Dict, Any


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(
        self, query: str, top_k: int = 5, filter_metadata: Dict[str, Any] = None
    ) -> List[Document]:
        """검색 쿼리 처리 메서드 (모든 Retriever에서 구현 필수)"""
        pass
