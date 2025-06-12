from langchain.text_splitter import CharacterTextSplitter
from .base_chunker import BaseChunker
from langchain.schema import Document
from typing import List


class FixedChunker(BaseChunker):
    def __init__(
        self, chunk_size: int = 1000, chunk_overlap: int = 200, separator="\n\n"
    ):
        self.splitter = CharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator=separator
        )

    def chunk(self, docs: List[Document]) -> List[Document]:
        return self.splitter.split_documents(docs)
