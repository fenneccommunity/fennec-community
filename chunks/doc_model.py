"""
نماذج البيانات الأساسية المحسّنة لنظام التقسيم الذكي
Enhanced data models for the AI-Powered Chunking System
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from enum import Enum
import hashlib
import uuid
import numpy as np


class ChunkType(str, Enum):
    """chunks types"""
    PARAGRAPH   = "paragraph"
    SENTENCE    = "sentence"
    SECTION     = "section"
    CODE_BLOCK  = "code_block"
    LIST_ITEM   = "list_item"
    HEADER      = "header"
    TABLE       = "table"
    SEMANTIC    = "semantic"
    ADAPTIVE    = "adaptive"
    STRUCTURAL  = "structural"
    WINDOW      = "window"


class DocumentType(str, Enum):
    """Document types"""
    PLAIN_TEXT  = "plain_text"
    MARKDOWN    = "markdown"
    HTML        = "html"
    PDF         = "pdf"
    ARABIC      = "arabic"
    MIXED       = "mixed"


@dataclass
class ChunkMetadata:
    """
    Rich metadata for every chunk
    """
    source: str = ""
    page: Optional[int] = None
    section: Optional[str] = None
    position: int = 0                   # ترتيب الـ chunk داخل المستند
    total_chunks: int = 0               # العدد الكلي للـ chunks في المستند
    chunk_type: ChunkType = ChunkType.PARAGRAPH
    document_type: DocumentType = DocumentType.PLAIN_TEXT
    language: str = "auto"
    keywords: List[str] = field(default_factory=list)
    score: float = 0.0                  # chunk importance score
    keyword_density: float = 0.0
    is_header: bool = False
    heading_level: int = 0              # 0 = not a header, 1-6 = H1-H6
    char_start: int = 0                 # character offset start in original
    char_end: int = 0                   # character offset end in original
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentChunk:
    """
    Core text unit carrying text, embedding, and rich metadata
    """
    text: str
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str = ""
    embedding: Optional[np.ndarray] = None
    metadata: ChunkMetadata = field(default_factory=ChunkMetadata)
    # Hash للـ deduplication
    _hash: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.text or not self.text.strip():
            raise ValueError("text is required and cannot be empty ")
        self._hash = self._compute_hash()

    def _compute_hash(self) -> str:
        normalized = " ".join(self.text.split()).lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @property
    def id(self) -> str:
        return self.chunk_id

    @property
    def content_hash(self) -> str:
        return self._hash

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id":   self.chunk_id,
            "doc_id":     self.doc_id,
            "text":       self.text,
            "char_count": self.char_count,
            "word_count": self.word_count,
            "hash":       self._hash,
            "has_embedding": self.embedding is not None,
            "metadata": {
                "source":          self.metadata.source,
                "page":            self.metadata.page,
                "section":         self.metadata.section,
                "position":        self.metadata.position,
                "total_chunks":    self.metadata.total_chunks,
                "chunk_type":      self.metadata.chunk_type.value,
                "document_type":   self.metadata.document_type.value,
                "language":        self.metadata.language,
                "keywords":        self.metadata.keywords,
                "score":           self.metadata.score,
                "keyword_density": self.metadata.keyword_density,
                "is_header":       self.metadata.is_header,
                "heading_level":   self.metadata.heading_level,
                "char_start":      self.metadata.char_start,
                "char_end":        self.metadata.char_end,
                "extra":           self.metadata.extra,
            }
        }

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return f"DocumentChunk(id={self.chunk_id[:8]}…, chars={self.char_count}, text='{preview}…')"


@dataclass
class Document:
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: str = field(default_factory=lambda: str(uuid.uuid4()))
