from .base_reranker import BaseReranker
from langchain.schema import Document
from typing import List
from FlagEmbedding import FlagReranker


class BgeReranker(BaseReranker):
    def __init__(
        self, model_name: str = "BAAI/bge-reranker-v2-m3", use_fp16: bool = True
    ):
        self.reranker = FlagReranker(model_name, use_fp16=use_fp16)

    def rerank(self, query: str, docs: List[Document], top_n: int) -> List[Document]:
        if not docs:
            return []
        pairs = [[query, doc.page_content] for doc in docs]
        # normalize=True로 0~1 점수 반환(시그모이드)
        scores = self.reranker.compute_score(pairs, normalize=True)
        # 점수와 문서를 묶어 정렬
        scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:top_n]]
