from .base_chunker import BaseChunker
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
import copy


class CustomChunker(BaseChunker):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", " ", ""],
            length_function=len,
        )

    def chunk(self, docs: List[Document]) -> List[Document]:
        all_chunks = []
        for doc in docs:
            metadata = copy.deepcopy(doc.metadata)
            if len(doc.page_content) > self.chunk_size:
                split_docs = self.splitter.split_documents([doc])
                for split_doc in split_docs:
                    split_doc.metadata = metadata
                all_chunks.extend(split_docs)
            else:
                all_chunks.append(
                    Document(page_content=doc.page_content, metadata=metadata)
                )
        return all_chunks
