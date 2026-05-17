from typing import List, Union, Optional, Dict, Any, Tuple
import numpy as np
import logging
import time
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import RLock
from collections import OrderedDict
from .base_embedder import BaseEmbedder
from .config_embedder import EmbedderConfig

logger = logging.getLogger(__name__)
config = EmbedderConfig()


@dataclass
class MistralUsageStats:
    """ Mistral usage statistics"""
    total_tokens: int = 0
    total_requests: int = 0
    total_cost: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    errors: int = 0
    cache_hits: int = 0

    def add_request(self, token_count: int, cost: float):
        """Add new request"""
        self.total_requests += 1
        self.total_tokens += token_count
        self.total_cost += cost

    def get_summary(self) -> Dict[str, Any]:
        """Get statistics summary"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        return {
            'total_tokens': self.total_tokens,
            'total_requests': self.total_requests,
            'total_cost_usd': round(self.total_cost, 6),
            'elapsed_time_seconds': round(elapsed, 2),
            'avg_tokens_per_request': self.total_tokens // max(1, self.total_requests),
            'errors': self.errors,
            'cache_hits': self.cache_hits,
        }


class MistralRateLimiter:
    """Rate limiter for Mistral API"""

    def __init__(self, max_requests_per_second: int = 5):
        """
        Initialize rate limiter

        Args:
            max_requests_per_second:  Max requests per second
        """
        self.max_requests_per_second = max_requests_per_second
        self.requests: List[datetime] = []
        self.lock = RLock()

    def wait_if_needed(self):
        """  Wait if rate limit would be exceeded"""
        with self.lock:
            now = datetime.now()
            cutoff = now - timedelta(seconds=1)
            self.requests = [r for r in self.requests if r > cutoff]

            if len(self.requests) >= self.max_requests_per_second:
                wait_time = 1.0 - (now - self.requests[0]).total_seconds()
                if wait_time > 0:
                    logger.info(
                        f"Waiting {wait_time:.2f}s to avoid rate limit"
                    )
                    time.sleep(wait_time)
                    self.requests = [r for r in self.requests if r > datetime.now() - timedelta(seconds=1)]

            self.requests.append(datetime.now())


class MistralEmbedder(BaseEmbedder):
    """
    Mistral AI Embeddings Interface with Arabic support
    Supports Mistral embedding models:
    - mistral-embed (1024 dimensions) 

    Features:
    - Native Arabic language support 
    - Rate limiting 
    - Cost tracking 
    - Retry logic with exponential backoff 
    - LRU caching 
    - Batch processing

    Examples:
        >>> embedder = MistralEmbedder(
        ...     model_name="mistral-embed",
        ...     api_key="your-api-key"
        ... )
        >>> embeddings = embedder.encode("مرحبا بك في عالم الذكاء الاصطناعي")

        >>> # Batch encoding | الترميز الجماعي
        >>> texts = ["النص الأول", "النص الثاني", "النص الثالث"]
        >>> embeddings = embedder.encode(texts)
    """

    # Model specifications | مواصفات النماذج
    MODEL_SPECS = {
        "mistral-embed": {
            "dimensions": 1024,
            "max_tokens": 8192,
            "cost_per_1m_tokens": 0.10,
            "arabic_support": "good",
            "release_date": "2024-01",
            "description": "Mistral's dedicated embedding model"
        }
    }

    # Default model | النموذج الافتراضي
    DEFAULT_MODEL = "mistral-embed"

    def __init__(
        self,
        model_name: str = "mistral-embed",
        api_key: Optional[str] = None,
        base_url: str = "https://api.mistral.ai/v1",
        normalize_embeddings: bool = True,
        batch_size: int = 50,
        max_length: Optional[int] = None,
        cache_embeddings: bool = True,
        cache_size: int = 5000,
        show_progress: bool = False,
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        enable_rate_limiting: bool = True,
        max_requests_per_second: int = 5,
        track_costs: bool = True,
        **kwargs
    ):
        """
        Initialize Mistral Embedder

        Args:
            model_name:  Mistral model name
            api_key:  API key For Mistral Model
            base_url:  Base API URL
            normalize_embeddings: Normalize embeddings
            batch_size:  Batch size for processing
            max_length:  Maximum text length (in tokens)
            cache_embeddings: Enable caching
            cache_size:  Cache size
            show_progress:  Show progress bar
            timeout:  Request timeout in seconds
            max_retries:  Maximum number of retries
            retry_delay:  Base retry delay (seconds)
            enable_rate_limiting:  Enable rate limiting
            max_requests_per_second:  Max requests per second
            track_costs:  Track API costs
            **kwargs:  Additional parameters
        """
        super().__init__(
            model_name=model_name,
            normalize_embeddings=normalize_embeddings,
            batch_size=batch_size,
            max_length=max_length,
            cache_embeddings=cache_embeddings,
            show_progress=show_progress,
            **kwargs
        )

        # Get API key | الحصول على مفتاح API
        self.api_key = api_key or os.getenv('MISTRAL_API_KEY')
        if not self.api_key:
            raise ValueError(
                "   Mistral API key required\n"
                "   Provide via api_key parameter or MISTRAL_API_KEY environment variable"
            )

        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.track_costs = track_costs

        # Validate model | التحقق من النموذج
        if model_name not in self.MODEL_SPECS:
            available_models = ', '.join(self.MODEL_SPECS.keys())
            logger.warning(
                f"   Model {model_name} unknown. Supported: {available_models}"
            )
            self.model_spec = {
                "dimensions": 1024,
                "max_tokens": 8192,
                "cost_per_1m_tokens": 0.10,
                "arabic_support": "unknown"
            }
        else:
            self.model_spec = self.MODEL_SPECS[model_name]

        # LRU cache | ذاكرة التخزين المؤقت
        self._lru_cache: OrderedDict = OrderedDict()
        self._cache_max_size = cache_size

        # Usage stats | إحصائيات الاستخدام
        if self.track_costs:
            self.usage_stats = MistralUsageStats()

        # Rate limiter | محدد المعدل
        if enable_rate_limiting:
            self.rate_limiter = MistralRateLimiter(
                max_requests_per_second=max_requests_per_second
            )
            logger.info(
                f"🚦 Rate limiter enabled: "
                f"{max_requests_per_second} req/s"
            )
        else:
            self.rate_limiter = None

        # Initialize Mistral client | تهيئة عميل Mistral
        try:
            from mistralai import Mistral
            self.client = Mistral(
                api_key=self.api_key,
                timeout_ms=timeout * 1000
            )
            logger.info("✅ Mistral client initialized")
        except ImportError:
            raise ImportError(
                "   mistralai library not installed\n"
                "   Install it: pip install mistralai"
            )

        # Validate API key | التحقق من صحة المفتاح
        self._validate_api_key()

        logger.info(
            f"✅   Mistral Embedder Initialized\n"
            f"   📌 Model: {model_name}\n"
            f"   🔢 Dimensions: {self.model_spec['dimensions']}\n"
            f"   📦 Batch size: {batch_size}\n"
            f"   💰 Cost per 1M tokens: "
            f"${self.model_spec['cost_per_1m_tokens']}\n"
            f"   🌍 Arabic support: {self.model_spec['arabic_support']}\n"
            f"   💾 Caching: "
            f"{'activate' if cache_embeddings else 'deactivated'} "
        )

    def _validate_api_key(self):
        """
        Validate API key with a minimal test request
        """
        try:
            logger.debug("🔐 Validating API key...")
            self.client.embeddings.create(
                model=self.model_name,
                inputs=["test"]
            )
            logger.info("✅ Mistral API key valid")
        except Exception as e:
            err_str = str(e).lower()
            if "unauthorized" in err_str or "401" in err_str or "api key" in err_str:
                raise ValueError(
                    f"   Invalid Mistral API key\n"
                    f"   Error: {e}"
                )
            # Other errors (network, etc.) are non-fatal at init
            logger.warning(f"⚠️ Warning during validation: {e}")

    def _cache_get(self, key: str) -> Optional[np.ndarray]:
        """Get value from LRU cache"""
        if key in self._lru_cache:
            self._lru_cache.move_to_end(key)
            return self._lru_cache[key]
        return None

    def _cache_set(self, key: str, value: np.ndarray):
        """Set value in LRU cache"""
        if key in self._lru_cache:
            self._lru_cache.move_to_end(key)
        self._lru_cache[key] = value
        if len(self._lru_cache) > self._cache_max_size:
            self._lru_cache.popitem(last=False)

    def _process_batch(self, texts: List[str]) -> Tuple[np.ndarray, int]:
        """
        Process a batch of texts with retry logic

        Args:
            texts: List of texts

        Returns:
            (Embeddings array, token count)
        """
        # Apply rate limiting | تطبيق تحديد المعدل
        if self.rate_limiter:
            self.rate_limiter.wait_if_needed()

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    inputs=texts
                )

                # Extract embeddings | استخراج التضمينات
                embeddings = [
                    np.array(item.embedding, dtype=np.float32)
                    for item in response.data
                ]

                # Get token usage | الحصول على عدد الرموز
                total_tokens = getattr(response.usage, 'total_tokens', 0)

                # Track usage | تتبع الاستخدام
                if self.track_costs:
                    cost = (total_tokens / 1_000_000) * self.model_spec['cost_per_1m_tokens']
                    self.usage_stats.add_request(total_tokens, cost)

                return np.array(embeddings, dtype=np.float32), total_tokens

            except Exception as e:
                last_error = str(e)
                err_str = last_error.lower()

                # Rate limit | تجاوز حد المعدل
                if "429" in err_str or "rate limit" in err_str:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Rate limit (attempt {attempt+1}/{self.max_retries}) — "
                        f"⏳ Waiting to avoid rate limit:{wait_time:.1f}s"
                    )
                    if attempt < self.max_retries - 1:
                        time.sleep(wait_time)
                    if self.track_costs:
                        self.usage_stats.errors += 1

                # Connection or server error — retryable
                elif "connection" in err_str or "timeout" in err_str or "5" in err_str[:3]:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Connection error: {last_error[:80]} — ⏳ {wait_time:.1f}s"
                    )
                    if attempt < self.max_retries - 1:
                        time.sleep(wait_time)

                # Non-retryable error | خطأ غير قابل للتكرار
                else:
                    logger.error(f"❌ Mistral API error: {last_error[:150]}")
                    if self.track_costs:
                        self.usage_stats.errors += 1
                    raise

        raise RuntimeError(
            f"Batch failed after {self.max_retries} retries. Last error: {last_error}"
        )

    def encode(
        self,
        texts: Union[str, List[str]],
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Convert texts to embeddings

        Args:
            texts: Single text or list of texts
            show_progress_bar: Show progress bar
            convert_to_numpy: Convert to numpy array
            batch_size: Batch size override
            **kwargs: Additional parameters

        Returns:
             Embeddings array of shape (N, 1024)
        """
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            raise ValueError("❌ Empty text list provided")

        # Filter and clean | تنظيف النصوص
        original_count = len(texts)
        texts = [t.strip() for t in texts if t and t.strip()]

        if len(texts) < original_count:
            logger.warning(
                f"Skipped {original_count - len(texts)} empty texts"
            )

        if not texts:
            raise ValueError("❌ All texts are empty after filtering")

        # Truncate if needed | اقتطاع النصوص الطويلة
        if self.max_length:
            texts = [t[:self.max_length] for t in texts]

        effective_batch_size = batch_size or self.batch_size
        start_time = time.time()
        all_embeddings: List[np.ndarray] = []
        total_texts = len(texts)
        total_tokens_used = 0
        cache_hits = 0

        show_progress = show_progress_bar or self.show_progress

        if show_progress:
            try:
                from tqdm import tqdm
                batch_iterator = tqdm(
                    range(0, total_texts, effective_batch_size),
                    desc="🔄 Generating embeddings",
                    unit="batch"
                )
            except ImportError:
                batch_iterator = range(0, total_texts, effective_batch_size)
        else:
            batch_iterator = range(0, total_texts, effective_batch_size)

        for i in batch_iterator:
            batch = texts[i:i + effective_batch_size]
            batch_to_embed: List[str] = []
            batch_indices: List[int] = []
            cached_map: Dict[int, np.ndarray] = {}

            # Check cache | فحص الذاكرة المؤقتة
            for j, text in enumerate(batch):
                if self.cache_embeddings:
                    cached = self._cache_get(text)
                    if cached is not None:
                        cached_map[j] = cached
                        cache_hits += 1
                        if self.track_costs:
                            self.usage_stats.cache_hits += 1
                        continue
                batch_to_embed.append(text)
                batch_indices.append(j)

            if batch_to_embed:
                batch_embeddings, tokens_used = self._process_batch(batch_to_embed)
                total_tokens_used += tokens_used

                # Cache new results | تخزين النتائج الجديدة
                if self.cache_embeddings:
                    for text, emb in zip(batch_to_embed, batch_embeddings):
                        self._cache_set(text, emb)

                # Merge cached + new | دمج النتائج
                dim = batch_embeddings.shape[1]
                final_batch = np.zeros((len(batch), dim), dtype=np.float32)
                for idx, emb in cached_map.items():
                    final_batch[idx] = emb
                for idx, emb_idx in enumerate(batch_indices):
                    final_batch[emb_idx] = batch_embeddings[idx]
            else:
                # Fully cached batch | دفعة مخزنة بالكامل
                dim = list(cached_map.values())[0].shape[0]
                final_batch = np.array(
                    [cached_map[j] for j in range(len(batch))],
                    dtype=np.float32
                )

            all_embeddings.append(final_batch)

        # Stack results | تجميع النتائج
        embeddings_array = np.vstack(all_embeddings)

        # Normalize | التطبيع
        if self.normalize_embeddings:
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            embeddings_array = embeddings_array / norms

        # Update stats | تحديث الإحصائيات
        elapsed = time.time() - start_time
        self._stats['total_encodings'] += 1
        self._stats['total_texts'] += total_texts
        self._stats['total_time'] += elapsed

        cache_hit_rate = (cache_hits / total_texts * 100) if total_texts > 0 else 0
        batch_cost = (total_tokens_used / 1_000_000) * self.model_spec['cost_per_1m_tokens']

        logger.info(
            f"✅  generated:  {total_texts}  Embeddings In: {elapsed:.2f}s\n"
            f"   📊 Avg: {elapsed/total_texts:.3f}s/text\n"
            f"   💾 Cache: {cache_hits}/{total_texts} ({cache_hit_rate:.1f}%)\n"
            f"   🔢 Tokens: {total_tokens_used:,}\n"
            f"   💰 Cost: ${batch_cost:.6f}\n"
            f"   📏 Dim: {self.embedding_dim}"
        )

        return embeddings_array

    @property
    def embedding_dim(self) -> int:
        """
        Get embedding dimension
        """
        return self.model_spec['dimensions']

    def estimate_cost(self, texts: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Estimate the cost of embedding the given texts

        Args:
            texts: Text or list of texts

        Returns:
            Cost information
        """
        if isinstance(texts, str):
            texts = [texts]

        # Estimate: ~1 token per 4 chars (works for Arabic too)
        total_chars = sum(len(t) for t in texts)
        estimated_tokens = total_chars // 4

        cost = (estimated_tokens / 1_000_000) * self.model_spec['cost_per_1m_tokens']
        return {
            "texts_count": len(texts),
            "estimated_tokens": estimated_tokens,
            "avg_tokens_per_text": estimated_tokens // max(1, len(texts)),
            "cost_per_1m_tokens_usd": self.model_spec['cost_per_1m_tokens'],
            "estimated_cost_usd": round(cost, 6),
            "model": self.model_name,
            "dimensions": self.embedding_dim,
        }

    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get detailed usage statistics
        """
        stats = self._stats.copy()
        stats['embedding_dim'] = self.embedding_dim
        stats['model_name'] = self.model_name
        stats['cache_enabled'] = self.cache_embeddings
        stats['cache_size'] = len(self._lru_cache)
        stats['cache_max_size'] = self._cache_max_size
        if stats['total_texts'] > 0:
            stats['cache_hit_rate'] = stats.get('cache_hits', 0) / stats['total_texts']
        else:
            stats['cache_hit_rate'] = 0.0
        if self.track_costs:
            stats['api_usage'] = self.usage_stats.get_summary()
        return stats

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model
        """
        info = self.model_spec.copy()
        info.update({
            'model_name': self.model_name,
            'normalize_embeddings': self.normalize_embeddings,
            'batch_size': self.batch_size,
            'cache_enabled': self.cache_embeddings,
            'rate_limiting_enabled': self.rate_limiter is not None,
            'cost_tracking_enabled': self.track_costs,
        })
        if self.track_costs:
            info['usage_stats'] = self.usage_stats.get_summary()
        return info

    def clear_cache(self):
        """
        Clear the embedding cache
        """
        size = len(self._lru_cache)
        self._lru_cache.clear()
        logger.info(
            f"🗑️ Cleared {size} embeddings from cache"
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.track_costs:
            summary = self.usage_stats.get_summary()
            logger.info(
                f"\n📊 Mistral Session Summary:\n"
                f"   🔢 Total tokens: {summary['total_tokens']:,}\n"
                f"   📝 Total requests: {summary['total_requests']}\n"
                f"   💰 Total cost: ${summary['total_cost_usd']:.6f}\n"
                f"   ❌ Errors: {summary['errors']}\n"
                f"   ⏱️ Elapsed: {summary['elapsed_time_seconds']:.1f}s"
            )

