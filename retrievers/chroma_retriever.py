from langchain_community.vectorstores import Chroma
from .base_retriever import BaseRetriever
from typing import List, Dict, Any
from langchain.schema import Document


class ChromaRetriever(BaseRetriever):
    def __init__(self, vectorstore: Chroma):
        self.vectorstore = vectorstore

    def retrieve(
        self, query: str, top_k: int = 5, filter_metadata: Dict[str, Any] = None
    ) -> List[Document]:
        """Chroma 기반 하이브리드 검색"""
        return self.vectorstore.similarity_search(
            query=query, k=top_k, filter=filter_metadata
        )
