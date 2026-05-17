"""
Removes duplicate chunks using fast hash-based exact matching or cosine similarity
"""
from __future__ import annotations
import logging
from typing import List, Optional, Set, Dict, Tuple

import numpy as np

from .doc_model import DocumentChunk
from .embeddings import EmbeddingProvider, cosine_similarity_matrix

logger = logging.getLogger(__name__)


class Deduplicator:
    """
    Applies two layers of deduplication:
    1. Hash-based: detects exact matches after text normalization.
    2. Similarity-based: detects near-duplicates using cosine similarity.
    """

    def __init__(
        self,
        use_hash: bool = True,
        use_similarity: bool = False,          # similarity-based needs embeddings (expensive)
        similarity_threshold: float = 0.95,
        embedding_model: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.use_hash = use_hash
        self.use_similarity = use_similarity
        self.similarity_threshold = similarity_threshold
        self._embed: Optional[EmbeddingProvider] = None

        if use_similarity and embedding_model:
            try:
                self._embed = EmbeddingProvider(model_name=embedding_model, device=device)
            except Exception as exc:
                logger.warning(f"[Deduplicator] Could not load embedding model: {exc}. Similarity dedup disabled.")
                self.use_similarity = False

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def deduplicate(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Returns deduplicated chunk list.
        """
        if not chunks:
            return chunks

        # Stage 1: hash dedup (fast, O(n))
        if self.use_hash:
            chunks = self._hash_dedup(chunks)
            logger.debug(f"[Deduplicator] After hash dedup: {len(chunks)} chunks")

        # Stage 2: similarity dedup (slower, O(n²) embeddings)
        if self.use_similarity and self._embed is not None:
            chunks = self._similarity_dedup(chunks)
            logger.debug(f"[Deduplicator] After similarity dedup: {len(chunks)} chunks")

        return chunks

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _hash_dedup(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        seen: Set[str] = set()
        result: List[DocumentChunk] = []
        for ch in chunks:
            h = ch.content_hash
            if h not in seen:
                seen.add(h)
                result.append(ch)
        removed = len(chunks) - len(result)
        if removed:
            logger.info(f"[Deduplicator] Removed {removed} exact duplicate(s)")
        return result

    def _similarity_dedup(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        للـ chunks التي لا يوجد لها embedding بعد، يحسبها ثم يُزيل المتشابهة جداً.
        Computes missing embeddings then removes near-duplicate chunks.
        """
        texts = [c.text for c in chunks]
        embeddings_list = self._embed.embed_batch(texts)
        embs = np.array(embeddings_list)

        sim_matrix = cosine_similarity_matrix(embs, embs)
        n = len(chunks)
        removed: Set[int] = set()

        for i in range(n):
            if i in removed:
                continue
            for j in range(i + 1, n):
                if j in removed:
                    continue
                if sim_matrix[i, j] >= self.similarity_threshold:
                    # Keep the longer chunk
                    keep = i if len(chunks[i].text) >= len(chunks[j].text) else j
                    drop = j if keep == i else i
                    removed.add(drop)

        result = [ch for idx, ch in enumerate(chunks) if idx not in removed]
        if removed:
            logger.info(f"[Deduplicator] Removed {len(removed)} near-duplicate chunk(s)")
        return result
