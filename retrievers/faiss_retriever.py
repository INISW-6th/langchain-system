from langchain_community.vectorstores import FAISS
from .base_retriever import BaseRetriever
from typing import List, Dict, Any
from langchain.schema import Document


class FAISSRetriever(BaseRetriever):
    def __init__(self, vectorstore: FAISS):
        self.vectorstore = vectorstore

    def retrieve(
        self, query: str, top_k: int = 5, filter_metadata: Dict[str, Any] = None
    ) -> List[Document]:
        """FAISS 기반 유사도 검색 + 메타데이터 필터링"""
        return self.vectorstore.similarity_search(
            query=query, k=top_k, filter=filter_metadata
        )
