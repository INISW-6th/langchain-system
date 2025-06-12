from langchain.schema import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from typing import Dict, List, Any, Optional
import torch
import copy

# ▼▼▼ 커스텀 모듈 임포트 ▼▼▼
from retrievers import FAISSRetriever, ChromaRetriever, WeaviateRetriever
from rerankers import BgeReranker, CohereReranker
from chunkers import MetadataChunkGenerator
from config import get_hf_llm


class ModularRAGExperiment:
    def __init__(self, config: Dict[str, Any], purpose_docs: Dict[str, List[Document]]):
        self.config = config
        self.purpose_docs = purpose_docs
        self.rag_components = self._initialize_components()
        self.vector_stores = self._build_vector_stores()

    def _initialize_components(self) -> Dict:
        """환경설정 기반 컴포넌트 초기화"""
        # 1. 청킹 설정
        chunk_config = self.config["chunking"]
        chunk_method = chunk_config["method"]

        if chunk_method == "recursive":
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_config["chunk_size"],
                chunk_overlap=chunk_config["chunk_overlap"],
                separators=chunk_config["separators"],
                length_function=len,
            )
        elif chunk_method == "fixed":
            splitter = CharacterTextSplitter(
                chunk_size=chunk_config["chunk_size"],
                chunk_overlap=chunk_config["chunk_overlap"],
                separator=chunk_config["separator"],
            )
        elif chunk_method == "custom":
            splitter = MetadataChunkGenerator(
                chunk_size=chunk_config["chunk_size"],
                chunk_overlap=chunk_config["chunk_overlap"],
            )
        else:
            raise ValueError(f"지원하지 않는 청킹 방법: {chunk_method}")

        # 2. 임베딩 설정
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedder = HuggingFaceEmbeddings(
            model_name=self.config["embedding"]["model_name"],
            model_kwargs={"device": device},
        )

        return {"splitter": splitter, "embedder": embedder}

    def _build_vector_stores(self) -> Dict[str, Any]:
        """목적별 벡터 저장소 구축"""
        vector_stores = {}
        for purpose, docs in self.purpose_docs.items():
            # 문서 분할
            if isinstance(self.rag_components["splitter"], MetadataChunkGenerator):
                chunks = self.rag_components["splitter"].generate_chunks(docs)
            else:
                chunks = self.rag_components["splitter"].split_documents(docs)

            # 벡터 저장소 생성
            vector_db_type = self.config["vector_db"]
            if vector_db_type == "faiss":
                vector_store = FAISS.from_documents(
                    chunks, self.rag_components["embedder"]
                )
                retriever = FAISSRetriever(vector_store)
            elif vector_db_type == "chroma":
                vector_store = Chroma.from_documents(
                    chunks, self.rag_components["embedder"]
                )
                retriever = ChromaRetriever(vector_store)
            elif vector_db_type == "weaviate":
                vector_store = WeaviateRetriever.from_documents(
                    chunks,
                    self.rag_components["embedder"],
                    self.config["weaviate_config"],
                )
            else:
                raise ValueError("지원하지 않는 벡터DB")

            vector_stores[purpose] = retriever
        return vector_stores

    def ask_modular_rag(self, purpose: str, question: str, prompt_template: str) -> str:
        """RAG 파이프라인 실행"""
        # 1. 검색
        retriever = self.vector_stores[purpose]
        docs = retriever.retrieve(question, top_k=self.config["initial_top_k"])

        # 2. 리랭킹
        reranker_type = self.config.get("reranker")
        if reranker_type == "bge":
            reranker = BgeReranker()
            docs = reranker.rerank(question, docs, self.config["rerank_top_k"])
        elif reranker_type == "cohere":
            reranker = CohereReranker(api_key=self.config["cohere_api_key"])
            docs = reranker.rerank(question, docs, self.config["rerank_top_k"])

        # 3. 프롬프트 생성
        context = "\n\n".join(doc.page_content for doc in docs)
        prompt = prompt_template.format(context=context, question=question)

        # 4. LLM 호출
        llm = get_hf_llm(
            self.config["llm"]["model_name"],
            system_prompt=self.config["llm"]["system_prompt"],
        )
        return llm(prompt)

    def clear_gpu_cache(self):
        """GPU 메모리 정리"""
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
