from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import asdict , is_dataclass
# Import DocumentChunk – adjust path if the package layout changes
try:
    from chunks.doc_model import DocumentChunk
except ImportError:
    try:
        from ..chunks.doc_model import DocumentChunk
    except ImportError:
        # Last-resort fallback so the file can be imported standalone
        from dataclasses import dataclass, field

        @dataclass
        class DocumentChunk:  # type: ignore[no-redef]
            chunk_id: str
            doc_id: str
            text: str
            embedding: Optional[np.ndarray] = None
            metadata: Dict = field(default_factory=dict)

            def __post_init__(self):
                if not self.chunk_id or not self.doc_id:
                    raise ValueError("chunk_id and doc_id are required")
                if not self.text or not self.text.strip():
                    raise ValueError("text cannot be empty")

            @property
            def id(self):
                return self.chunk_id


logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def sanitize_metadata(metadata: Dict) -> Dict:
    """
    Sanitize a metadata dictionary so its values are safe for *all* backends.

    Rules
    -----
    - str / int / float / bool / None  → kept as-is
    - list[str]                        → kept (used by Pinecone $in filter)
    - list (mixed)                     → joined as comma-separated string
    - dict / set / anything else       → JSON-serialised string

    This is the *single* place where metadata sanitisation lives;
    every backend calls this helper so the logic is never duplicated.
    """
    import json
    if is_dataclass(metadata):
        metadata = asdict(metadata)
    clean: Dict = {}
    for key, value in metadata.items():
        if value is None:
            # ChromaDB rejects None values — skip the key entirely.
            # Other backends (FAISS, Pinecone) are also fine without it.
            continue
        elif isinstance(value, (str, int, float, bool)):
            clean[key] = value
        elif isinstance(value, list):
            # ChromaDB rejects list values entirely — always join to a string.
            # Pinecone $in filters can be re-parsed from this string if needed.
            clean[key] = ", ".join(str(item) for item in value)
        else:
            try:
                clean[key] = json.dumps(value, ensure_ascii=False)
            except Exception:
                clean[key] = str(value)
    return clean


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base class
# ─────────────────────────────────────────────────────────────────────────────

class VectorDatabaseBase(ABC):
    """
    Abstract base class for all vector database backends.

    Every subclass **must** implement:
        • add
        • search
        • remove_by_doc_id
        • get_stats
        • __len__

    Every subclass **should** implement (optional, sensible default provided):
        • clear
        • save / load
        • get_chunk_by_id
        • get_chunks_by_doc_id
        • asearch / aadd  (async wrappers)

    RAGSystem only calls the five mandatory methods plus the optional
    async wrappers. No backend-specific names leak into orchestration code.
    """

    # ── Constructor helpers ───────────────────────────────────────────────────

    def __init__(self, embedder: Optional[Any] = None) -> None:
        """
        Base initializer – stores the embedder that is shared by all backends.

        Args:
            embedder: Any object with an `.encode(texts: List[str]) -> np.ndarray`
                      method (e.g. SentenceTransformer, GeminiEmbedder …).
                      Pass ``None`` when you will always supply pre-computed
                      embeddings.
        """
        self.embedder = embedder

    # ── Embedding helpers ─────────────────────────────────────────────────────

    def _require_embedder(self) -> None:
        """Raise if no embedder is set."""
        if self.embedder is None:
            raise ValueError(
                "An embedder is required for this operation. "
                "Pass one in the constructor or supply pre-computed embeddings."
            )

    def _encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts into an embedding matrix.

        Returns:
            np.ndarray of shape (len(texts), embedding_dim), dtype float32
        """
        self._require_embedder()
        try:
            result = self.embedder.encode(texts)
            return np.asarray(result, dtype=np.float32)
        except Exception as exc:
            raise RuntimeError(f"Embedding generation failed: {exc}") from exc

    def _encode_query(self, query: Union[str, np.ndarray]) -> np.ndarray:
        """
        Return a 1-D float32 array for *query*, encoding it if it is a string.
        """
        if isinstance(query, str):
            return self._encode([query])[0]
        return np.asarray(query, dtype=np.float32)

    # ── Mandatory abstract interface ──────────────────────────────────────────

    @abstractmethod
    def add(
        self,
        chunks: List[DocumentChunk],
        embeddings: Optional[np.ndarray] = None,
    ) -> None:
        """
        Add (or upsert) a list of document chunks.

        Args:
            chunks:     List of DocumentChunk objects.
            embeddings: Optional pre-computed embeddings (shape: N × dim, float32).
                        If None the backend will call ``self._encode(texts)``.

        Subclass notes
        ──────────────
        • Chroma  – internally converts chunks to (ids, documents, metadatas).
        • FAISS   – adds vectors to the FAISS index and updates lookup tables.
        • Pinecone– upserts vectors in batches of UPSERT_BATCH_SIZE.

        All backends must call ``sanitize_metadata(chunk.metadata)`` before
        storing metadata to ensure cross-backend compatibility.
        """

    @abstractmethod
    def search(
        self,
        query: Union[str, np.ndarray],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for the most similar chunks to *query*.

        Args:
            query:           Text string or pre-computed 1-D embedding vector.
            top_k:           Maximum number of results to return.
            score_threshold: Minimum similarity score; results below this are
                             discarded.  Pass ``None`` to disable filtering.
            **kwargs:        Backend-specific parameters (e.g. ``filter_dict``
                             for Pinecone, ``filters`` for Chroma,
                             ``doc_id_filter`` for FAISS).

        Returns:
            List of ``(DocumentChunk, float)`` tuples sorted by score
            descending (highest similarity first).

        Unified return type
        ───────────────────
        Chroma previously returned ``(text_str, score, metadata_dict)``.
        After inheriting from this base, Chroma's ``search`` also returns
        ``(DocumentChunk, float)`` so RAGSystem never needs isinstance checks.
        """

    @abstractmethod
    def remove_by_doc_id(self, doc_id: str) -> int:
        """
        Remove all chunks that belong to *doc_id*.

        Unified name
        ────────────
        • FAISS   used ``remove_by_doc_id``    ← same name, no change needed
        • Pinecone used ``delete_by_doc_id``   ← alias added in subclass
        • Chroma  used ``delete_by_filter``    ← alias added in subclass

        Returns:
            Number of chunks removed (0 if doc_id was not found).
        """

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Return a dictionary of database statistics.
        Unified name
        ────────────
        • FAISS / Pinecone / Chroma used ``get_stats()``  ← same, no change

        Every backend must include at least these keys:
            total_vectors   : int   – total number of stored vectors
            embedding_dim   : int   – dimension of each vector
            distance_metric : str   – e.g. 'cosine'
            has_embedder    : bool  – whether an embedder is configured
        """

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of stored vectors."""

    # ── Optional methods with sensible defaults ───────────────────────────────

    def clear(self) -> int:
        """
        Remove *all* chunks from the database.

        Returns:
            Number of chunks that were removed.

        Default behaviour: raises NotImplementedError.
        Backends that support clearing should override this.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement clear(). "
            "Override this method in the subclass."
        )

    def get_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        """
        Retrieve a single chunk by its ID from local cache.
        Default behaviour: returns None (no local cache).
        Backends with a local chunk store should override this.
        """
        return None

    def get_chunks_by_doc_id(self, doc_id: str) -> List[DocumentChunk]:
        """
        Retrieve all chunks that belong to *doc_id* from local cache.
        Default behaviour: returns [] (no local cache).
        """
        return []

    def save(self, path: Union[str, Path]) -> None:
        """
        Persist the database to disk.
        Default behaviour: no-op with a warning (cloud backends like
        Pinecone are always persistent and don't need a local save).
        """
        logger.warning(
            "%s.save() called but this backend does not support "
            "local persistence. Data is managed remotely.",
            type(self).__name__,
        )

    @classmethod
    def load(cls, path: Union[str, Path], **kwargs: Any) -> "VectorDatabaseBase":
        """
        Load a previously saved database from disk.
        Default behaviour: raises NotImplementedError.
        """
        raise NotImplementedError(
            f"{cls.__name__} does not implement load(). "
            "Override this classmethod in the subclass."
        )

    # ── Async wrappers (default: run sync version in a thread) ───────────────

    async def asearch(
        self,
        query: Union[str, np.ndarray],
        top_k: int = 10,
        score_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Async version of :meth:`search`.
        Runs the synchronous ``search`` inside a thread pool so it never
        blocks the event loop.  Backends with native async support should
        override this.
        """
        import asyncio
        return await asyncio.to_thread(
            self.search, query, top_k, score_threshold, **kwargs
        )

    async def aadd(
        self,
        chunks: List[DocumentChunk],
        embeddings: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> None:
        """
        Async version of :meth:`add`.
        """
        import asyncio
        return await asyncio.to_thread(self.add, chunks, embeddings, **kwargs)

    async def aremove_by_doc_id(self, doc_id: str) -> int:
        """
        Async version of :meth:`remove_by_doc_id`.
        """
        import asyncio
        return await asyncio.to_thread(self.remove_by_doc_id, doc_id)

    # ── Context-manager support ───────────────────────────────────────────────

    async def __aenter__(self) -> "VectorDatabaseBase":
        return self

    async def __aexit__(self, *_: Any) -> bool:
        return False

    def __enter__(self) -> "VectorDatabaseBase":
        return self

    def __exit__(self, *_: Any) -> bool:
        return False

    # ── Representation ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"vectors={len(self)}, "
            f"embedder={type(self.embedder).__name__ if self.embedder else None})"
        )
    

