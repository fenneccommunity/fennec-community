"""
Full OpenAI Embeddings support with Arabic language optimizations
This module provides integration with OpenAI's embedding models
"""

from typing import List, Union, Optional, Dict, Any, Tuple
import numpy as np
import logging
import time
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import tiktoken
from threading import RLock
from .base_embedder import BaseEmbedder
from .config_embedder import EmbedderConfig

logger = logging.getLogger(__name__)
config = EmbedderConfig()


@dataclass
class UsageStats:
    """Usage statistics"""
    total_tokens: int = 0
    total_requests: int = 0
    total_cost: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    requests_by_minute: Dict[str, int] = field(default_factory=dict)
    
    def add_request(self, tokens: int, cost: float):
        """Add new request"""
        self.total_tokens += tokens
        self.total_requests += 1
        self.total_cost += cost
        
        # Track requests per minute | تتبع الطلبات بالدقيقة
        minute_key = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.requests_by_minute[minute_key] = self.requests_by_minute.get(minute_key, 0) + 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get statistics summary"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        return {
            'total_tokens': self.total_tokens,
            'total_requests': self.total_requests,
            'total_cost_usd': round(self.total_cost, 6),
            'elapsed_time_seconds': round(elapsed, 2),
            'avg_tokens_per_request': self.total_tokens // max(1, self.total_requests),
            'requests_per_minute': len(self.requests_by_minute),
        }


class RateLimiter:
    """Rate limiter for API requests"""
    
    def __init__(self, max_requests_per_minute: int = 3000, max_tokens_per_minute: int = 1_000_000):
        """
        Initialize rate limiter
        Args:
          max_requests_per_minute: Maximum number of requests per minute
          max_tokens_per_minute: Maximum number of tokens per minute

        """
        self.max_requests_per_minute = max_requests_per_minute
        self.max_tokens_per_minute = max_tokens_per_minute
        self.requests = []
        self.tokens = []
        self.lock = RLock()
    
    def _clean_old_entries(self, entries: List[datetime]) -> List[datetime]:
        """Remove old entries"""
        cutoff = datetime.now() - timedelta(minutes=1)
        return [entry for entry in entries if entry > cutoff]
    
    def wait_if_needed(self, estimated_tokens: int):
        """
        Wait if rate limit would be exceeded
        
        Args:
            estimated_tokens: Estimated tokens
        """
        with self.lock:
            self.requests = self._clean_old_entries(self.requests)
            self.tokens = self._clean_old_entries(self.tokens)
            
            # Check request limit | فحص حد الطلبات
            if len(self.requests) >= self.max_requests_per_minute:
                wait_time = 60 - (datetime.now() - self.requests[0]).total_seconds()
                if wait_time > 0:
                    logger.info(f"⏳ Waiting to avoid rate limit")
                    time.sleep(wait_time)
                    self.requests = self._clean_old_entries(self.requests)
            
            # Check token limit | فحص حد الرموز
            current_tokens = len(self.tokens)
            if current_tokens + estimated_tokens > self.max_tokens_per_minute:
                wait_time = 60 - (datetime.now() - self.tokens[0]).total_seconds()
                if wait_time > 0:
                    logger.info(f"⏳ Waiting to avoid token limit")
                    time.sleep(wait_time)
                    self.tokens = self._clean_old_entries(self.tokens)
            
            # Record this request | تسجيل هذا الطلب
            now = datetime.now()
            self.requests.append(now)
            for _ in range(estimated_tokens):
                self.tokens.append(now)


class OpenAIEmbedder(BaseEmbedder):
    """
    OpenAI Embeddings Interface with Arabic support    
    Supports all OpenAI embedding models including:
    - text-embedding-3-large (Best quality, 3072 dimensions)
    - text-embedding-3-small (Balanced performance, 1536 dimensions)
    - text-embedding-ada-002 (Legacy, 1536 dimensions)
    All models support Arabic language natively
    
    Features:
    - Automatic rate limiting 
    - Token counting with tiktoken 
    - Cost tracking 
    - Retry logic with exponential backoff 
    - Dimension reduction support 
    
    Examples:
        >>> embedder = OpenAIEmbedder(
        ...     model_name="text-embedding-3-large",
        ...     api_key="your-api-key"
        ... )
        >>> embeddings = embedder.encode("مرحبا بك في عالم الذكاء الاصطناعي")
        
        >>> # Batch encoding with dimension reduction
        >>> embedder = OpenAIEmbedder(
        ...     model_name="text-embedding-3-large",
        ...     dimensions=1024  # Reduce from 3072 to 1024
        ... )
        >>> texts = ["النص الأول", "النص الثاني", "النص الثالث"]
        >>> embeddings = embedder.encode(texts)
    """
    
    # Model specifications | مواصفات النماذج
    MODEL_SPECS = {
        "text-embedding-3-large": {
            "max_dimensions": 3072,
            "default_dimensions": 3072,
            "max_tokens": 8191,
            "cost_per_1m_tokens": 0.13,
            "arabic_support": "excellent",
            "release_date": "2024-01"
        },
        "text-embedding-3-small": {
            "max_dimensions": 1536,
            "default_dimensions": 1536,
            "max_tokens": 8191,
            "cost_per_1m_tokens": 0.02,
            "arabic_support": "excellent",
            "release_date": "2024-01"
        },
        "text-embedding-ada-002": {
            "max_dimensions": 1536,
            "default_dimensions": 1536,
            "max_tokens": 8191,
            "cost_per_1m_tokens": 0.10,
            "arabic_support": "good",
            "release_date": "2022-12"
        }
    }
    
    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        dimensions: Optional[int] = None,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        batch_size: int = 100,  # OpenAI supports up to 2048
        max_length: Optional[int] = None,
        cache_embeddings: bool = True,
        show_progress: bool = False,
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        enable_rate_limiting: bool = True,
        max_requests_per_minute: int = 3000,
        max_tokens_per_minute: int = 1_000_000,
        track_costs: bool = True,
        **kwargs
    ):
        """
        Initialize OpenAI Embedder
        
        Args:
            model_name:  OpenAI model name
            api_key:  API key For openai
            dimensions: Number of dimensions (for reduction)
            device:  Device (not used for API)
            normalize_embeddings:  Normalize embeddings
            batch_size:  Batch size for processing
            max_length:  Maximum text length (in tokens)
            cache_embeddings: Enable caching
            show_progress:  Show progress bar
            timeout: Request timeout in seconds
            max_retries:  Maximum number of retries
            retry_delay:  Base retry delay (seconds)
            enable_rate_limiting:  Enable rate limiting
            max_requests_per_minute:  Max requests per minute
            max_tokens_per_minute:  Max tokens per minute
            track_costs: Track API costs
            **kwargs:  Additional parameters
        """
        super().__init__(
            model_name=model_name,
            device=device,
            normalize_embeddings=normalize_embeddings,
            batch_size=batch_size,
            max_length=max_length,
            cache_embeddings=cache_embeddings,
            show_progress=show_progress,
            **kwargs
        )
        
        # Get API key | الحصول على مفتاح API
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError(
                "   OpenAI API key required\n"
                "   Provide via api_key parameter or OPENAI_API_KEY environment variable"
            )
        
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.track_costs = track_costs
        
        # Initialize usage stats | تهيئة إحصائيات الاستخدام
        if self.track_costs:
            self.usage_stats = UsageStats()
        
        # Validate model | التحقق من النموذج
        if model_name not in self.MODEL_SPECS:
            available_models = ', '.join(self.MODEL_SPECS.keys())
            logger.warning(
                f"   Model {model_name} unknown\n"
                f"   Supported: {available_models}"
            )
            # Use default specs | استخدام المواصفات الافتراضية
            self.model_spec = {
                "max_dimensions": 1536,
                "default_dimensions": 1536,
                "max_tokens": 8191,
                "cost_per_1m_tokens": 0.10,
                "arabic_support": "unknown"
            }
        else:
            self.model_spec = self.MODEL_SPECS[model_name]
        
        # Set dimensions | تحديد الأبعاد
        max_dims = self.model_spec["max_dimensions"]
        
        if dimensions is not None:
            if dimensions > max_dims:
                logger.warning(
                    f"   Requested dimensions ({dimensions}) exceed max ({max_dims})\n"
                    f"   Using {max_dims}"
                )
                dimensions = max_dims
            elif dimensions < 1:
                raise ValueError(f"❌Dimensions must be > 0: {dimensions}")
            self.dimensions = dimensions
        else:
            self.dimensions = self.model_spec["default_dimensions"]
        
        # Initialize rate limiter | تهيئة محدد المعدل
        if enable_rate_limiting:
            self.rate_limiter = RateLimiter(
                max_requests_per_minute=max_requests_per_minute,
                max_tokens_per_minute=max_tokens_per_minute
            )
            logger.info(
                f"🚦 Rate limiter enabled\n"
                f"   📊 Requests/min: {max_requests_per_minute}\n"
                f"   🔢 Tokens/min: {max_tokens_per_minute:,}"
            )
        else:
            self.rate_limiter = None
        
        # Initialize tokenizer for accurate token counting
        # تهيئة محلل الرموز لعد دقيق
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            logger.debug("✅ Tokenizer loaded")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load tokenizer: {e}")
            self.tokenizer = None
        
        # Initialize OpenAI client | تهيئة عميل OpenAI
        try:
            from openai import OpenAI, APIError, RateLimitError, APIConnectionError
            self.client = OpenAI(
                api_key=self.api_key,
                timeout=timeout,
                max_retries=0  # نحن سنتعامل مع إعادة المحاولة | We handle retries
            )
            # Store exception classes | حفظ فئات الاستثناءات
            self.APIError = APIError
            self.RateLimitError = RateLimitError
            self.APIConnectionError = APIConnectionError
            
            logger.info("✅ OpenAI client initialized")
        except ImportError:
            raise ImportError(
                "   openai library not installed\n"
                "   Install it: pip install openai"
            )
        
        # Validate API key | التحقق من صحة المفتاح
        self._validate_api_key()
        
        logger.info(
            f"✅ OpenAI Embedder Initialized\n"
            f"   📌 Model: {model_name}\n"
            f"   🔢 Dimensions: {self.dimensions}\n"
            f"   📦 Batch size: {batch_size}\n"
            f"   💰 Cost per 1M tokens: ${self.model_spec['cost_per_1m_tokens']}\n"
            f"   🌍 Arabic support: {self.model_spec['arabic_support']}\n"
            f"   💾 Caching: {'active' if cache_embeddings else 'unactive'}\n"
            f"   🚦 Rate limiting: {'active' if enable_rate_limiting else 'unactive'}"
        )
    
    def _validate_api_key(self):
        """
        Validate API key by making a test request
        """
        try:
            logger.debug("🔐 Validating API key...")
            # Make a minimal test request | طلب تجريبي بسيط
            test_response = self.client.embeddings.create(
                input="test",
                model=self.model_name,
                dimensions=self.dimensions if self.model_name.startswith("text-embedding-3") else None
            )
            logger.info("✅ API key valid")
        except self.APIError as e:
            logger.error(f"❌ Invalid API key: {e}")
            raise ValueError(
                f"   Invalid API key or model unavailable\n"
                f"   Error: {e}"
            )
        except Exception as e:
            logger.warning(f"⚠️ Warning during validation: {e}")
    
    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text accurately
        
        Args:
            text: النص | Text to count tokens for
            
        Returns:
            عدد الرموز | Number of tokens
        """
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception as e:
                logger.debug(f"⚠️ Token counting failed: {e}")
        
        # Fallback: estimate tokens | احتياطي: تقدير الرموز
        # For Arabic: ~1 token per 4 characters | للعربية: ~1 رمز لكل 4 أحرف
        return len(text) // 4
    
    def _process_batch(self, texts: List[str]) -> Tuple[np.ndarray, int]:
        """
        Process a batch of texts with retry logic        
        Args:
            texts: List of texts
            
        Returns:
            (Array of embeddings, token count)
        """
        # Count tokens | عد الرموز
        total_tokens = sum(self._count_tokens(text) for text in texts)
        
        # Apply rate limiting | تطبيق تحديد المعدل
        if self.rate_limiter:
            self.rate_limiter.wait_if_needed(total_tokens)
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                # Prepare request parameters | تحضير معاملات الطلب
                request_params = {
                    "input": texts,
                    "model": self.model_name
                }
                
                # Add dimensions if model supports it
                # إضافة الأبعاد إذا كان النموذج يدعمها
                if self.model_name.startswith("text-embedding-3"):
                    request_params["dimensions"] = self.dimensions
                
                # Make API call | إجراء استدعاء API
                response = self.client.embeddings.create(**request_params)
                
                # Extract embeddings | استخراج التضمينات
                embeddings = [np.array(item.embedding, dtype=np.float32) 
                             for item in response.data]
                
                # Get actual token count from response | الحصول على عدد الرموز الفعلي
                actual_tokens = response.usage.total_tokens
                
                # Track usage | تتبع الاستخدام
                if self.track_costs:
                    cost = (actual_tokens / 1_000_000) * self.model_spec['cost_per_1m_tokens']
                    self.usage_stats.add_request(actual_tokens, cost)
                
                return np.array(embeddings), actual_tokens
                
            except self.RateLimitError as e:
                last_error = f"🚦 Rate limit exceeded"
                wait_time = self.retry_delay * (2 ** attempt)
                logger.warning(
                    f"{last_error} (retry {attempt + 1}/{self.max_retries})\n"
                    f"   ⏳ waiting {wait_time:.1f}s"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(wait_time)
                    
            except self.APIConnectionError as e:
                last_error = f"🔌 Connection error: {str(e)[:100]}"
                logger.warning(f"{last_error} (retry {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
                    
            except self.APIError as e:
                last_error = f"❌ API error: {str(e)[:100]}"
                logger.error(f"{last_error}")
                # Don't retry on API errors | لا تعيد المحاولة عند أخطاء API
                raise
                
            except Exception as e:
                last_error = f"❌ Unexpected error: {str(e)[:100]}"
                logger.error(f"{last_error}")
                raise
        
        # All retries failed | فشلت جميع المحاولات
        error_msg = f"❌  batch processing failed after {self.max_retries} retries\n   last error: {last_error}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    def encode(
        self,
        texts: Union[str, List[str]],
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Convert texts to embeddings with optimized processing
        
        Args:
            texts: Single text or list of texts
            show_progress_bar: Show progress bar
            convert_to_numpy: Convert to numpy array
            batch_size:  Batch size override
            **kwargs: Additional parameters
            
        Returns:
            Array of embeddings
        """
        # Convert single text to list | تحويل النص الواحد إلى قائمة
        if isinstance(texts, str):
            texts = [texts]
        
        # Validate input | التحقق من المدخلات
        if not texts:
            raise ValueError("❌ Empty text list provided")
        
        # Filter empty texts | تصفية النصوص الفارغة
        original_count = len(texts)
        texts = [text.strip() for text in texts if text and text.strip()]
        
        if len(texts) < original_count:
            logger.warning(
                f"   Ignored {original_count - len(texts)} empty texts"
            )
        
        if not texts:
            raise ValueError("❌ All texts are empty")
        
        start_time = time.time()
        all_embeddings = []
        total_texts = len(texts)
        total_tokens_used = 0
        
        # Use provided batch_size or default | استخدام حجم الدفعة المحدد أو الافتراضي
        effective_batch_size = batch_size or self.batch_size
        
        # Progress tracking | تتبع التقدم
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
                logger.warning("⚠️ tqdm not installed. Install for progress: pip install tqdm")
        else:
            batch_iterator = range(0, total_texts, effective_batch_size)
        
        # Process in batches | المعالجة على دفعات
        cache_hits = 0
        cache_misses = 0
        
        for i in batch_iterator:
            batch = texts[i:i + effective_batch_size]
            batch_texts_to_process = []
            batch_indices = []
            cached_embeddings = {}
            
            # Check cache for each text | التحقق من الذاكرة المؤقتة
            for j, text in enumerate(batch):
                if self.cache_embeddings and text in self._cache:
                    cached_embeddings[j] = self._cache[text]
                    cache_hits += 1
                    self._stats['cache_hits'] += 1
                else:
                    batch_texts_to_process.append(text)
                    batch_indices.append(j)
                    cache_misses += 1
            
            # Process non-cached texts | معالجة النصوص غير المخزنة
            if batch_texts_to_process:
                try:
                    batch_embeddings, tokens_used = self._process_batch(batch_texts_to_process)
                    total_tokens_used += tokens_used
                    
                    # Cache new embeddings | تخزين التضمينات الجديدة
                    if self.cache_embeddings:
                        for text, embedding in zip(batch_texts_to_process, batch_embeddings):
                            self._cache[text] = embedding
                    
                    # Combine cached and new embeddings | دمج التضمينات المخزنة والجديدة
                    final_batch_embeddings = np.zeros(
                        (len(batch), batch_embeddings.shape[1]),
                        dtype=np.float32
                    )
                    
                    # Fill in cached embeddings | ملء التضمينات المخزنة
                    for idx, emb in cached_embeddings.items():
                        final_batch_embeddings[idx] = emb
                    
                    # Fill in new embeddings | ملء التضمينات الجديدة
                    for idx, emb_idx in enumerate(batch_indices):
                        final_batch_embeddings[emb_idx] = batch_embeddings[idx]
                        
                except Exception as e:
                    logger.error(f"❌ failed to process batch: {i//effective_batch_size + 1}: {e}")
                    raise
            else:
                # All from cache | كلها من الذاكرة المؤقتة
                final_batch_embeddings = np.array(
                    [cached_embeddings[j] for j in range(len(batch))],
                    dtype=np.float32
                )
            
            all_embeddings.append(final_batch_embeddings)
        
        # Combine all batches | دمج جميع الدفعات
        embeddings_array = np.vstack(all_embeddings)
        
        # Normalize if requested | التطبيع إذا طُلب
        if self.normalize_embeddings:
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            # Avoid division by zero | تجنب القسمة على صفر
            norms = np.where(norms == 0, 1, norms)
            embeddings_array = embeddings_array / norms
        
        # Update statistics | تحديث الإحصائيات
        elapsed_time = time.time() - start_time
        self._stats['total_encodings'] += 1
        self._stats['total_texts'] += total_texts
        self._stats['total_time'] += elapsed_time
        self._stats['avg_time_per_text'] = self._stats['total_time'] / self._stats['total_texts']
        
        # Calculate cache efficiency | حساب كفاءة التخزين المؤقت
        cache_hit_rate = (cache_hits / total_texts * 100) if total_texts > 0 else 0
        
        # Calculate cost | حساب التكلفة
        if total_tokens_used > 0:
            batch_cost = (total_tokens_used / 1_000_000) * self.model_spec['cost_per_1m_tokens']
        else:
            batch_cost = 0
        
        logger.info(
            f"✅ Generated: {total_texts} \n"
            f"   📊 Avg time: {elapsed_time/total_texts:.3f}s/text\n"
            f"   💾 Cache: {cache_hits}/{total_texts} ({cache_hit_rate:.1f}%)\n"
            f"   🔢 Tokens used: {total_tokens_used:,}\n"
            f"   💰 Cost: ${batch_cost:.6f}\n"
            f"   📏 Dimension: {self.embedding_dim}"
        )
        
        return embeddings_array
    
    @property
    def embedding_dim(self) -> int:
        """
        Get embedding dimension
        
        Returns:
            Embedding vector dimension
        """
        return self.dimensions
    
    def estimate_cost(self, texts: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Estimate the cost of embedding the given texts
        
        Args:
            texts: Text or list of texts
            
        Returns:
            Cost information dictionary
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Count tokens accurately | عد الرموز بدقة
        total_tokens = sum(self._count_tokens(text) for text in texts)
        
        cost_per_1m = self.model_spec['cost_per_1m_tokens']
        estimated_cost = (total_tokens / 1_000_000) * cost_per_1m
        
        return {
            "texts_count": len(texts),
            "total_tokens": total_tokens,
            "avg_tokens_per_text": total_tokens // max(1, len(texts)),
            "cost_per_1m_tokens_usd": cost_per_1m,
            "estimated_cost_usd": round(estimated_cost, 6),
            "model": self.model_name,
            "dimensions": self.dimensions
        }
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get detailed usage statistics
        
        Returns:
            Usage statistics dictionary
        """
        stats = self._stats.copy()
        stats['embedding_dim'] = self.embedding_dim
        stats['model_name'] = self.model_name
        stats['cache_enabled'] = self.cache_embeddings
        stats['cache_size'] = len(self._cache) if self.cache_embeddings else 0
        
        # Calculate cache hit rate | حساب معدل إصابة الكاش
        if stats['total_texts'] > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['total_texts']
        else:
            stats['cache_hit_rate'] = 0.0
        
        # Add API usage stats if tracking | إضافة إحصائيات API إذا كانت مفعلة
        if self.track_costs:
            stats['api_usage'] = self.usage_stats.get_summary()
        
        return stats
    
    def clear_cache(self):
        """
        Clear the embedding cache
        """
        cache_size = len(self._cache)
        self._cache.clear()
        logger.info(f"🗑️ Cleared {cache_size} embeddings from cache")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model
        
        Returns:
            Model information
        """
        info = self.model_spec.copy()
        info.update({
            'model_name': self.model_name,
            'configured_dimensions': self.dimensions,
            'normalize_embeddings': self.normalize_embeddings,
            'batch_size': self.batch_size,
            'cache_enabled': self.cache_embeddings,
            'rate_limiting_enabled': self.rate_limiter is not None,
            'cost_tracking_enabled': self.track_costs
        })
        
        if self.track_costs:
            info['usage_stats'] = self.usage_stats.get_summary()
        
        return info
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with summary"""
        if self.track_costs:
            summary = self.usage_stats.get_summary()
            logger.info(
                f"\n📊 Session Summary:\n"
                f"   🔢 Total tokens: {summary['total_tokens']:,}\n"
                f"   📝 Total requests: {summary['total_requests']}\n"
                f"   💰 Total cost: ${summary['total_cost_usd']:.6f}\n"
                f"   ⏱️ Elapsed: {summary['elapsed_time_seconds']:.1f}s"
            )

