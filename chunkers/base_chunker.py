from abc import ABC, abstractmethod
from langchain.schema import Document
from typing import List


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, docs: List[Document]) -> List[Document]:
        """문서를 청킹(분할)하는 메서드 (하위 클래스에서 구현 필수)"""
        pass
