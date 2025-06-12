from .base_reranker import BaseReranker
from langchain.schema import Document
from typing import List
import cohere


class CohereReranker(BaseReranker):
    def __init__(self, api_key: str, model_name: str = "rerank-english-v2.0"):
        self.client = cohere.Client(api_key)
        self.model_name = model_name

    def rerank(self, query: str, docs: List[Document], top_n: int) -> List[Document]:
        if not docs:
            return []
        response = self.client.rerank(
            model=self.model_name,
            query=query,
            documents=[doc.page_content for doc in docs],
            top_n=top_n,
        )
        # Cohere는 relevance_score로 정렬된 결과 반환
        return [docs[result.index] for result in response.results]
