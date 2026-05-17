from typing import List, Union, Optional, Dict, Any, Tuple
import numpy as np
import logging
from functools import wraps
import time
import hashlib
from dataclasses import dataclass
from enum import Enum
from .base_embedder import BaseEmbedder
from .config_embedder import EmbedderConfig
logger = logging.getLogger(__name__)
config = EmbedderConfig()


class ArabicQuality(Enum):
    """Arabic language support quality levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    UNKNOWN = "unknown"


@dataclass
class ModelInfo:
    """Model information and metadata"""
    full_name: str
    dimensions: int
    max_tokens: int
    arabic_quality: ArabicQuality
    size: str
    description: str = ""


@dataclass
class EmbeddingStats:
    """Statistics for embedding operations"""
    total_encodings: int = 0
    total_texts: int = 0
    total_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0

    @property
    def avg_time_per_text(self) -> float:
        return self.total_time / self.total_texts if self.total_texts > 0 else 0.0

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return (self.cache_hits / total * 100) if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_encodings": self.total_encodings,
            "total_texts": self.total_texts,
            "total_time": round(self.total_time, 2),
            "avg_time_per_text": round(self.avg_time_per_text, 4),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": round(self.cache_hit_rate, 2),
        }


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator for retrying failed operations\n
    مُزخرف لإعادة محاولة العمليات الفاشلة
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"   Attempt {attempt + 1}/{max_retries} failed, "
                            f"retrying in {current_delay:.1f}s...\n"
                            f"   Error: {str(e)}"
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"   All {max_retries + 1} attempts failed")

            raise last_exception
        return wrapper
    return decorator


class HuggingFaceEmbedder(BaseEmbedder):
    """
    HuggingFace Local Embeddings Interface with Arabic support
    Runs models locally via sentence-transformers — no API key needed.

    Recommended Arabic models:
        - sentence-transformers/paraphrase-multilingual-mpnet-base-v2  (768d)
        - sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2  (384d)
        - intfloat/multilingual-e5-large                               (1024d)
        - BAAI/bge-m3                                                  (1024d)
        - Alibaba-NLP/gte-multilingual-base                            (768d)

    Examples:
        >>> embedder = HuggingFaceEmbedder()
        >>> emb = embedder.encode("مرحبا بك في عالم الذكاء الاصطناعي")
        >>> print(emb.shape)   # (1, 384)

        >>> texts = ["النص الأول", "النص الثاني"]
        >>> embeddings = embedder.encode(texts)
        >>> print(embeddings.shape)   # (2, 384)
    """

    ARABIC_MODELS: Dict[str, ModelInfo] = {
        "paraphrase-multilingual-mpnet-base-v2": ModelInfo(
            full_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            dimensions=768,
            max_tokens=512,
            arabic_quality=ArabicQuality.EXCELLENT,
            size="420MB",
            description="Excellent multilingual model with strong Arabic support",
        ),
        "paraphrase-multilingual-MiniLM-L12-v2": ModelInfo(
            full_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            dimensions=384,
            max_tokens=512,
            arabic_quality=ArabicQuality.GOOD,
            size="118MB",
            description="Lightweight model with good Arabic support",
        ),
        "multilingual-e5-large": ModelInfo(
            full_name="intfloat/multilingual-e5-large",
            dimensions=1024,
            max_tokens=512,
            arabic_quality=ArabicQuality.EXCELLENT,
            size="2.24GB",
            description="High-quality multilingual embeddings",
        ),
        "bge-m3": ModelInfo(
            full_name="BAAI/bge-m3",
            dimensions=1024,
            max_tokens=8192,
            arabic_quality=ArabicQuality.EXCELLENT,
            size="2.27GB",
            description="State-of-the-art multilingual model with long context",
        ),
        "gte-multilingual-base": ModelInfo(
            full_name="Alibaba-NLP/gte-multilingual-base",
            dimensions=768,
            max_tokens=8192,
            arabic_quality=ArabicQuality.EXCELLENT,
            size="1.3GB",
            description="High-performance multilingual model",
        ),
    }

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        max_length: Optional[int] = None,
        cache_embeddings: bool = True,
        cache_ttl: Optional[int] = None,
        show_progress: bool = False,
        max_retries: int = 3,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        """
        Initialize HuggingFace Local Embedder

        Args:
            model_name: HuggingFace model name
            device: Device to use (cuda/cpu/mps)
            normalize_embeddings:  Normalize embeddings
            batch_size: Batch size for processing
            max_length:  Maximum text length
            cache_embeddings:  Enable caching
            cache_ttl:  Cache TTL in seconds (None = forever)
            show_progress:  Show progress bar
            max_retries:  Maximum retry attempts
            trust_remote_code:  Allow remote code execution
        """
        super().__init__(
            model_name=model_name,
            device=device,
            normalize_embeddings=normalize_embeddings,
            batch_size=batch_size,
            max_length=max_length,
            cache_embeddings=cache_embeddings,
            show_progress=show_progress,
            **kwargs,
        )

        self.max_retries = max(0, max_retries)
        self.trust_remote_code = trust_remote_code
        self.cache_ttl = cache_ttl
        self._embedding_dim: Optional[int] = None
        self._stats = EmbeddingStats()
        self._cache_timestamps: Dict[str, float] = {}

        self._load_model()
        self._log_initialization()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    @retry_on_failure(max_retries=3, delay=1.0, backoff=2.0)
    def _load_model(self) -> None:
        """
        Load local model via sentence-transformers.
        """
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"📥 Loading model: {self.model_name}")

            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                trust_remote_code=self.trust_remote_code,
            )

            self._embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"✅ Model loaded | 🔢 dim={self._embedding_dim}")

        except ImportError as e:
            raise ImportError(
                "❌ sentence-transformers not installed.\n"
                "   Install it:  pip install sentence-transformers"
            ) from e
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise RuntimeError(f"Failed to load model '{self.model_name}'") from e

    def _log_initialization(self) -> None:
        """Log initialization details."""
        info = self.get_model_info()
        logger.info(
            f"\n{'='*60}\n"
            f"✅  HuggingFace Embedder Initialized\n"
            f"{'='*60}\n"
            f"📌 Model      : {self.model_name}\n"
            f"🖥️  Device     : {self.device}\n"
            f"🔢 Dimensions : {self._embedding_dim}\n"
            f"📏 Max tokens : {info.get('max_tokens', 'Unknown')}\n"
            f"📦 Batch size : {self.batch_size}\n"
            f"🌍 Arabic     : {info.get('arabic_quality', 'Unknown')}\n"
            f"💾 Caching    : {'Enabled' if self.cache_embeddings else 'Disabled'}\n"
            f"🔄 Retries    : {self.max_retries}\n"
            f"{'='*60}"
        )

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _generate_cache_key(self, text: str) -> str:
        if len(text) > 100:
            return hashlib.md5(text.encode("utf-8")).hexdigest()
        return text

    def _is_cache_valid(self, cache_key: str) -> bool:
        if not self.cache_ttl:
            return True
        if cache_key not in self._cache_timestamps:
            return False
        return (time.time() - self._cache_timestamps[cache_key]) < self.cache_ttl

    def _get_from_cache(self, text: str) -> Optional[np.ndarray]:
        if not self.cache_embeddings:
            return None
        key = self._generate_cache_key(text)
        if key in self._cache and self._is_cache_valid(key):
            self._stats.cache_hits += 1
            return self._cache[key]
        self._stats.cache_misses += 1
        return None

    def _add_to_cache(self, text: str, embedding: np.ndarray) -> None:
        if not self.cache_embeddings:
            return
        key = self._generate_cache_key(text)
        self._cache[key] = embedding
        self._cache_timestamps[key] = time.time()

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def _process_batch(self, texts: List[str]) -> Tuple[np.ndarray, int, int]:
        """
        Process a batch of texts, using cache where possible.

        Returns:
            (embeddings, cached_count, new_count)
        """
        result_list: List[Tuple[int, np.ndarray]] = []
        texts_to_encode: List[str] = []
        indices_to_encode: List[int] = []
        cached_count = 0

        for i, text in enumerate(texts):
            cached = self._get_from_cache(text)
            if cached is not None:
                result_list.append((i, cached))
                cached_count += 1
            else:
                texts_to_encode.append(text)
                indices_to_encode.append(i)

        new_count = 0
        if texts_to_encode:
            new_embeddings = self.model.encode(
                texts_to_encode,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,  # we normalize ourselves later
            )
            for text, emb, orig_idx in zip(texts_to_encode, new_embeddings, indices_to_encode):
                self._add_to_cache(text, emb)
                result_list.append((orig_idx, emb))
                new_count += 1

        result_list.sort(key=lambda x: x[0])
        embeddings = np.array([emb for _, emb in result_list], dtype=np.float32)
        return embeddings, cached_count, new_count

    # ------------------------------------------------------------------
    # encode  (required by BaseEmbedder)
    # ------------------------------------------------------------------

    def encode(
        self,
        texts: Union[str, List[str]],
        show_progress_bar: Optional[bool] = None,
        convert_to_numpy: bool = True,
        normalize: Optional[bool] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Convert texts to embeddings.

        Args:
            texts:  Single text or list of texts
            show_progress_bar:  Show progress bar
            convert_to_numpy:  Always numpy
            normalize:  Override default normalization
            **kwargs:  Extra params (ignored)

        Returns:
            np.ndarray — shape (n, dim)
        """
        if not texts:
            raise ValueError("❌ Empty input provided")

        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]

        if not all(isinstance(t, str) for t in texts):
            raise ValueError("❌ All elements must be strings")

        # Drop empty strings
        texts = [t for t in texts if t.strip()]
        if not texts:
            raise ValueError("❌ All texts are empty after stripping whitespace")

        start_time = time.time()
        total = len(texts)
        show_progress = show_progress_bar if show_progress_bar is not None else self.show_progress
        should_normalize = normalize if normalize is not None else self.normalize_embeddings

        all_embeddings: List[np.ndarray] = []
        total_cached = 0
        total_new = 0

        for i in range(0, total, self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_embs, cached, new = self._process_batch(batch)
            all_embeddings.append(batch_embs)
            total_cached += cached
            total_new += new

            if show_progress:
                done = min(i + self.batch_size, total)
                logger.info(
                    f"🔄 Progress: {done}/{total} "
                    f"(💾 {total_cached} cached, 🆕 {total_new} new)"
                )

        embeddings_array = (
            np.vstack(all_embeddings) if len(all_embeddings) > 1 else all_embeddings[0]
        )

        if should_normalize:
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            embeddings_array = embeddings_array / (norms + 1e-10)

        elapsed = time.time() - start_time
        self._stats.total_encodings += 1
        self._stats.total_texts += total
        self._stats.total_time += elapsed

        logger.info(
            f"✅ {total} text(s) | ⏱️ {elapsed:.2f}s "
            f"({elapsed/total:.3f}s/text) | "
            f"💾 {total_cached} cached | 🆕 {total_new} new | "
            f"🔢 shape={embeddings_array.shape}"
        )

        return embeddings_array

    # ------------------------------------------------------------------
    # embedding_dim property  (required by BaseEmbedder)
    # ------------------------------------------------------------------

    @property
    def embedding_dim(self) -> int:
        if self._embedding_dim is None:
            self._embedding_dim = self.model.get_sentence_embedding_dimension()
        return self._embedding_dim

    # ------------------------------------------------------------------
    # Public utilities
    # ------------------------------------------------------------------

    def get_model_info(self) -> Dict[str, Any]:
        """ Model information."""
        for short_name, info in self.ARABIC_MODELS.items():
            if short_name in self.model_name or info.full_name == self.model_name:
                return {
                    "full_name": info.full_name,
                    "dimensions": info.dimensions,
                    "max_tokens": info.max_tokens,
                    "arabic_quality": info.arabic_quality.value,
                    "size": info.size,
                    "description": info.description,
                }
        return {
            "full_name": self.model_name,
            "dimensions": self._embedding_dim or "Unknown",
            "max_tokens": self.max_length or 512,
            "arabic_quality": ArabicQuality.UNKNOWN.value,
            "size": "Unknown",
            "description": "Custom model",
        }

    def get_stats(self) -> Dict[str, Any]:
        """ Performance statistics."""
        return self._stats.to_dict()

    def clear_cache(self) -> int:
        """ Clear the embedding cache. Returns number of cleared items."""
        count = len(self._cache)
        self._cache.clear()
        self._cache_timestamps.clear()
        logger.info(f"🧹 Cleared {count} items from cache")
        return count

    def __repr__(self) -> str:
        return (
            f"HuggingFaceEmbedder(\n"
            f"  model='{self.model_name}',\n"
            f"  device='{self.device}',\n"
            f"  dim={self._embedding_dim or 'Unknown'},\n"
            f"  batch_size={self.batch_size}\n"
            f")"
        )