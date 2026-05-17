# -*- coding: utf-8 -*-
"""
Base Document Loader  (improved)
الفئة الأساسية لتحميل المستندات (محسّنة)
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Iterator
from pathlib import Path
import logging
import hashlib
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class LoadedDocument:
    """
    Represents a loaded document with its content and metadata.
    """
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: Optional[str] = None

    def __post_init__(self):
        """
        Generate a DETERMINISTIC doc_id from content + source + page.
        FIX: The original used time.time() which made the same document
        produce a different ID on every run, breaking deduplication and
        caching. Now purely content-based → stable across runs.
    
        """
        if not self.doc_id:
            source = self.metadata.get("source", "")
            page = str(self.metadata.get("page_number", ""))
            seed = f"{source}:{page}:{self.page_content}"
            self.doc_id = "doc_" + hashlib.sha256(seed.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {"doc_id": self.doc_id, "page_content": self.page_content, "metadata": self.metadata}

    # --- NEW: useful dunder methods ---
    def __len__(self) -> int:
        """Character count of conten"""
        return len(self.page_content)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LoadedDocument):
            return NotImplemented
        return self.doc_id == other.doc_id

    def __hash__(self) -> int:
        return hash(self.doc_id)

    def __repr__(self) -> str:
        preview = self.page_content[:80].replace("\n", " ")
        return f"LoadedDocument(doc_id={self.doc_id!r}, chars={len(self)}, preview={preview!r}...)"


class BaseDocumentLoader(ABC):
    """Abstract base for all loaders"""

    def __init__(self, encoding: str = "utf-8", **kwargs):
        self.encoding = encoding
        self._extra_config = kwargs

    @abstractmethod
    def load(self) -> List[LoadedDocument]:
        pass

    def lazy_load(self) -> Iterator[LoadedDocument]:
        """Default: delegates to load(). Override for true streaming."""
        yield from self.load()

    def load_and_split(self, text_splitter=None) -> List[LoadedDocument]:
        documents = self.load()
        if text_splitter is None:
            return documents
        chunked = []
        for doc in documents:
            chunks = text_splitter.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                meta = doc.metadata.copy()
                meta.update({"chunk_index": i, "total_chunks": len(chunks), "original_doc_id": doc.doc_id})
                chunked.append(LoadedDocument(page_content=chunk, metadata=meta))
        return chunked

    def _build_metadata(self, source: str, **extra) -> Dict[str, Any]:
        meta = {"source": source, "loader_type": self.__class__.__name__, "loaded_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        meta.update(extra)
        return meta

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(encoding={self.encoding!r})"


class BaseFileLoader(BaseDocumentLoader):
    """Base for file-based loaders """

    SUPPORTED_EXTENSIONS: List[str] = []

    def __init__(self, file_path: str, encoding: str = "utf-8", **kwargs):
        super().__init__(encoding=encoding, **kwargs)
        self.file_path = Path(file_path)
        self._validate_file()

    def _validate_file(self):
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        if not self.file_path.is_file():
            raise ValueError(f"Path is not a file: {self.file_path}")
        if self.SUPPORTED_EXTENSIONS:
            ext = self.file_path.suffix.lower()
            if ext not in self.SUPPORTED_EXTENSIONS:
                raise ValueError(f"Unsupported extension '{ext}'. Supported: {self.SUPPORTED_EXTENSIONS}")

    def _build_file_metadata(self, **extra) -> Dict[str, Any]:
        stat = self.file_path.stat()
        meta = self._build_metadata(
            source=str(self.file_path.resolve()),
            file_name=self.file_path.name,
            file_type=self.file_path.suffix.lower(),
            file_size_bytes=stat.st_size,
        )
        meta.update(extra)
        return meta

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(file_path={str(self.file_path)!r})"
