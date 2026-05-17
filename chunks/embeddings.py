"""
Embedding provider with LRU caching and batch processing
"""
from __future__ import annotations
import logging
import hashlib
from functools import lru_cache
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity بين متجهين"""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Cosine similarity matrix 
    Returns shape (len(A), len(B))
    """
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-10)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-10)
    return A_norm @ B_norm.T


class EmbeddingProvider:
    """
    Provides embeddings using sentence-transformers with:
    - LRU cache per unique text
    - Batch processing for speed
    - TF-IDF fallback when sentence-transformers is unavailable
    """

    _instances: dict = {}   # singleton per model_name

    def __new__(cls, model_name: str, device: Optional[str] = None, cache_size: int = 10_000):
        key = (model_name, device)
        if key not in cls._instances:
            obj = super().__new__(cls)
            cls._instances[key] = obj
        return cls._instances[key]

    def __init__(self, model_name: str, device: Optional[str] = None, cache_size: int = 10_000) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self.model_name = model_name
        self.device = device or "cpu"
        self.cache_size = cache_size
        self._model = None
        self._use_fallback = False
        self._cache: dict = {}   # text_hash → np.ndarray
        self._dim: Optional[int] = None
        self._load_model()

    # ------------------------------------------------------------------ #
    # Model loading                                                        #
    # ------------------------------------------------------------------ #

    def _load_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._dim = self._model.get_sentence_embedding_dimension()
            logger.info(f"[EmbeddingProvider] Loaded '{self.model_name}' (dim={self._dim}) on {self.device}")
        except ImportError:
            logger.warning(
                "[EmbeddingProvider] sentence-transformers not installed. "
                "Falling back to TF-IDF embeddings. Install via: pip install sentence-transformers"
            )
            self._use_fallback = True
            self._dim = 256
        except Exception as exc:
            logger.warning(f"[EmbeddingProvider] Could not load '{self.model_name}': {exc}. Using TF-IDF fallback.")
            self._use_fallback = True
            self._dim = 256

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    @property
    def dim(self) -> int:
        return self._dim or 256

    def embed(self, text: str) -> np.ndarray:
        """Embed نص واحد مع caching"""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str], batch_size: int = 64) -> List[np.ndarray]:
        """
        Embed قائمة من النصوص مع caching لكل نص.
        Texts already in cache are not re-computed.
        """
        results: List[Optional[np.ndarray]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        for i, t in enumerate(texts):
            h = self._text_hash(t)
            if h in self._cache:
                results[i] = self._cache[h]
            else:
                uncached_indices.append(i)
                uncached_texts.append(t)

        if uncached_texts:
            embeddings = self._compute_embeddings(uncached_texts, batch_size)
            for idx, text, emb in zip(uncached_indices, uncached_texts, embeddings):
                h = self._text_hash(text)
                # Evict oldest entry if cache full
                if len(self._cache) >= self.cache_size:
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
                self._cache[h] = emb
                results[idx] = emb

        return results  # type: ignore[return-value]

    def pairwise_similarity(self, a: str, b: str) -> float:
        """Cosine similarity بين نصين"""
        ea, eb = self.embed_batch([a, b])
        return _cosine_similarity(ea, eb)

    def adjacent_similarities(self, sentences: List[str]) -> List[float]:
        """
        حساب cosine similarity بين كل جملة والتي تليها.
        Returns list of length len(sentences)-1.
        """
        if len(sentences) < 2:
            return []
        embs = np.array(self.embed_batch(sentences))
        sims = []
        for i in range(len(embs) - 1):
            sims.append(_cosine_similarity(embs[i], embs[i + 1]))
        return sims

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _compute_embeddings(self, texts: List[str], batch_size: int) -> List[np.ndarray]:
        if self._use_fallback:
            return self._tfidf_embeddings(texts)
        batches: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vecs = self._model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            batches.extend(vecs)
        return batches

    def _tfidf_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """TF-IDF fallback using hash-bucketing for fixed-dim output (no vocab mismatch)."""
        import re
        from collections import Counter

        vocab_size = self._dim  # always 256 — consistent across all calls

        def tokenize(t: str) -> List[str]:
            return re.findall(r'\w+', t.lower())

        def text_to_vec(tokens: List[str]) -> np.ndarray:
            vec = np.zeros(vocab_size, dtype=np.float32)
            counts = Counter(tokens)
            total = max(len(tokens), 1)
            for w, c in counts.items():
                idx = hash(w) % vocab_size  # deterministic hash bucket
                vec[idx] += c / total
            norm = np.linalg.norm(vec)
            return vec / (norm + 1e-10)

        return [text_to_vec(tokenize(t)) for t in texts]

    @staticmethod
    def _text_hash(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()
