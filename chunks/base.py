"""
الفئات الأساسية المجردة لنظام التقسيم الذكي
Abstract base classes for the AI-Powered Chunking System
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Callable, Optional, Dict, Any
from .doc_model import DocumentChunk, Document, ChunkMetadata


class BaseChunker(ABC):

    def chunk(self, text: str, doc_id: str = "", source: str = "") -> List[DocumentChunk]:
        """
        Public interface: split then post-process.
        """
        raw = self._chunk_impl(text, doc_id=doc_id, source=source)
        return self._finalize(raw, total=len(raw))

    @abstractmethod
    def _chunk_impl(self, text: str, doc_id: str, source: str) -> List[DocumentChunk]:
        """التنفيذ الفعلي لكل subclass"""
        ...

    # ------------------------------------------------------------------ #
    # Shared helpers                                                       #
    # ------------------------------------------------------------------ #

    def _finalize(self, chunks: List[DocumentChunk], total: int) -> List[DocumentChunk]:
        """ضبط position و total_chunks لكل chunk"""
        for i, ch in enumerate(chunks):
            ch.metadata.position = i
            ch.metadata.total_chunks = total
        return chunks

    def chunk_documents(self, documents: List[Document]) -> List[DocumentChunk]:
        """split a list of Documents into chunks, preserving doc_id and source metadata"""
        result: List[DocumentChunk] = []
        for doc in documents:
            source = doc.metadata.get("source", doc.doc_id)
            result.extend(self.chunk(doc.page_content, doc_id=doc.doc_id, source=source))
        return result


class TextSplitter(ABC):


    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        ...

    def split_documents(self, documents: List[Document]) -> List[Document]:
        texts = [d.page_content for d in documents]
        metas = [d.metadata for d in documents]
        return self.create_documents(texts, metas)

    def create_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Document]:
        metadatas = metadatas or [{}] * len(texts)
        docs: List[Document] = []
        for text, meta in zip(texts, metadatas):
            for chunk in self.split_text(text):
                docs.append(Document(page_content=chunk, metadata=meta.copy()))
        return docs


class ChunkingStrategy(ABC):
    """Legacy interface — kept for backward compatibility"""

    @abstractmethod
    def chunk(self, text: str, doc_id: str) -> List[DocumentChunk]:
        ...
