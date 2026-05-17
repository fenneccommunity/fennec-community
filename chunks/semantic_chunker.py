"""
Splits text by detecting semantic breaks using sentence embeddings + cosine similarity
"""
from __future__ import annotations
import re
import logging
import uuid
from typing import List, Optional

from .base import BaseChunker
from .doc_model import DocumentChunk, ChunkMetadata, ChunkType, DocumentType
from .chunk_config import ChunkConfig
from .embeddings import EmbeddingProvider

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────
# Sentence tokenizer (no NLTK dependency required)
# ────────────────────────────────────────────────────────────────────────

# Arabic sentence terminators
_ARABIC_SENT_RE = re.compile(
    r'(?<=[.!?؟\u06D4])\s+|(?<=\n)\n+'
)
# General multi-language sentence splitter
_SENT_RE = re.compile(
    r'(?<=[.!?؟。！？।\u06D4])\s+|\n{2,}'
)


def _split_sentences(text: str) -> List[str]:
    """
    Split text into sentences without requiring NLTK.
    """
    # Protect abbreviations like "Dr." / "No."
    protected = re.sub(r'\b([A-Z][a-z]{1,3})\.\s', r'\1@@@ ', text)
    parts = _SENT_RE.split(protected)
    sentences = [s.replace('@@@', '.').strip() for s in parts if s.strip()]
    return sentences


class SemanticChunker(BaseChunker):
    """
    Splits text into semantically coherent chunks by:
    1. Splitting text into sentences.
    2. Computing per-sentence embeddings.
    3. Computing adjacent cosine similarities.
    4. Opening a new chunk when similarity drops below threshold.
    5. Merging chunks that are too short.
    """

    def __init__(
        self,
        config: Optional[ChunkConfig] = None,
        *,
        similarity_threshold: float = 0.75,
        min_chunk_size: int = 50,
        max_chunk_size: int = 2048,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        language: str = "auto",
    ) -> None:
        cfg = config or ChunkConfig()
        self.similarity_threshold = similarity_threshold if config is None else cfg.semantic_similarity_threshold
        self.min_chunk_size = min_chunk_size if config is None else cfg.min_chunk_size
        self.max_chunk_size = max_chunk_size if config is None else cfg.max_chunk_size
        self.language = language

        _model = model_name or cfg.semantic_model
        self._embed = EmbeddingProvider(
            model_name=_model,
            device=device,
            cache_size=cfg.embedding_cache_size,
        )
        self._batch_size = cfg.embedding_batch_size

    # ------------------------------------------------------------------ #
    # BaseChunker implementation                                           #
    # ------------------------------------------------------------------ #

    def _chunk_impl(self, text: str, doc_id: str, source: str) -> List[DocumentChunk]:
        if not text or not text.strip():
            return []

        sentences = _split_sentences(text)
        if not sentences:
            return []

        # Single sentence → one chunk
        if len(sentences) == 1:
            return [self._make_chunk(sentences[0], doc_id=doc_id, source=source, position=0)]

        # Compute embeddings
        embeddings = self._embed.embed_batch(sentences, batch_size=self._batch_size)

        # Compute adjacent similarities
        from .embeddings import _cosine_similarity
        import numpy as np
        sims: List[float] = []
        for i in range(len(embeddings) - 1):
            sims.append(_cosine_similarity(
                np.array(embeddings[i]),
                np.array(embeddings[i + 1])
            ))

        # Group sentences into raw chunks based on similarity drops
        raw_groups: List[List[str]] = [[sentences[0]]]
        for i, sim in enumerate(sims):
            current_chunk_text = " ".join(raw_groups[-1])
            if sim < self.similarity_threshold or len(current_chunk_text) >= self.max_chunk_size:
                raw_groups.append([sentences[i + 1]])
            else:
                raw_groups[-1].append(sentences[i + 1])

        # Merge chunks that are too short
        merged: List[str] = self._merge_short_groups(raw_groups)

        # Build DocumentChunks
        chunks: List[DocumentChunk] = []
        for pos, chunk_text in enumerate(merged):
            if chunk_text.strip():
                chunks.append(self._make_chunk(
                    chunk_text.strip(),
                    doc_id=doc_id,
                    source=source,
                    position=pos,
                ))
        return chunks

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _merge_short_groups(self, groups: List[List[str]]) -> List[str]:
        """دمج المجموعات القصيرة مع التالية"""
        texts = [" ".join(g) for g in groups]
        result: List[str] = []
        buffer = ""
        for t in texts:
            if len(buffer) + len(t) < self.min_chunk_size:
                buffer = (buffer + " " + t).strip()
            else:
                if buffer:
                    result.append(buffer)
                buffer = t
        if buffer:
            result.append(buffer)
        return result

    def _make_chunk(self, text: str, doc_id: str, source: str, position: int) -> DocumentChunk:
        return DocumentChunk(
            text=text,
            chunk_id=str(uuid.uuid4()),
            doc_id=doc_id,
            metadata=ChunkMetadata(
                source=source,
                position=position,
                chunk_type=ChunkType.SEMANTIC,
                document_type=DocumentType.PLAIN_TEXT,
                language=self.language,
                char_start=0,
                char_end=len(text),
            ),
        )
