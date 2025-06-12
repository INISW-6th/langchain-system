from langchain_text_splitters import RecursiveCharacterTextSplitter
from .base_chunker import BaseChunker
from langchain.schema import Document
from typing import List


class RecursiveChunker(BaseChunker):
    def __init__(
        self, chunk_size: int = 1000, chunk_overlap: int = 200, separators=None
    ):
        if separators is None:
            separators = ["\n\n", "\n", "。", " ", ""]
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
        )

    def chunk(self, docs: List[Document]) -> List[Document]:
        return self.splitter.split_documents(docs)
