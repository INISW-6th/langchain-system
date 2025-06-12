import weaviate
from langchain_community.vectorstores.weaviate import Weaviate as WeaviateVectorStore
from .base_retriever import BaseRetriever
from typing import List, Dict, Any
from langchain.schema import Document


class WeaviateRetriever(BaseRetriever):
    def __init__(self, client: weaviate.Client, index_name: str):
        self.client = client
        self.index_name = index_name

    def retrieve(
        self, query: str, top_k: int = 5, filter_metadata: Dict[str, Any] = None
    ) -> List[Document]:
        """Weaviate 하이브리드 검색"""
        vectorstore = WeaviateVectorStore(
            client=self.client, index_name=self.index_name, text_key="content"
        )
        return vectorstore.similarity_search(
            query=query, k=top_k, filter=filter_metadata
        )
