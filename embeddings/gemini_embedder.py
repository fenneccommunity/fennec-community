from typing import List, Union, Optional, Dict, Any, Tuple
import numpy as np
import logging
import time
import os
from dataclasses import dataclass, field
from datetime import datetime
from .base_embedder import BaseEmbedder
from .config_embedder import EmbedderConfig

logger = logging.getLogger(__name__)
config = EmbedderConfig()


@dataclass
class GeminiUsageStats:
    """Gemini usage statistics"""
    total_requests: int = 0
    total_texts: int = 0
    total_characters: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    errors: int = 0
    cache_hits: int = 0
    
    def add_request(self, text_count: int, char_count: int):
        """Add new request"""
        self.total_requests += 1
        self.total_texts += text_count
        self.total_characters += char_count
    
    def add_error(self):
        """ Record error"""
        self.errors += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get statistics summary"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        return {
            'total_requests': self.total_requests,
            'total_texts': self.total_texts,
            'total_characters': self.total_characters,
            'elapsed_time_seconds': round(elapsed, 2),
            'avg_texts_per_request': self.total_texts / max(1, self.total_requests),
            'avg_chars_per_text': self.total_characters / max(1, self.total_texts),
            'errors': self.errors,
            'cache_hits': self.cache_hits,
            'success_rate': (self.total_requests - self.errors) / max(1, self.total_requests)
        }


class GeminiEmbedder(BaseEmbedder):
    """
    Google Gemini Embeddings Interface with Arabic support    
    ⚠️ IMPORTANT: Uses the NEW google-genai SDK (NOT google-generativeai)
    
    Installation:
        pip install google-genai
    
    Supports Gemini embedding models including:
    - gemini-embedding-001 (Latest, 3072 dimensions, best quality)
    - text-embedding-004 (Legacy, 768 dimensions, deprecated)
    - embedding-001 (Legacy, 768 dimensions, deprecated)
    All models support Arabic and 100+ languages natively    
    Features:
    - Task-specific embeddings 
    - Dimension reduction support (MRL) 
    - Smart retry logic 
    - Comprehensive error handling 
    - Usage tracking 
    
    Examples:
        >>> embedder = GeminiEmbedder(
        ...     model_name="gemini-embedding-001",
        ...     api_key="your-api-key"
        ... )
        >>> embeddings = embedder.encode("مرحبا بك في عالم الذكاء الاصطناعي")
        
        >>> # Batch encoding with task type
        >>> embedder = GeminiEmbedder(
        ...     model_name="gemini-embedding-001",
        ...     task_type="RETRIEVAL_DOCUMENT"
        ... )
        >>> texts = ["النص الأول", "النص الثاني", "النص الثالث"]
        >>> embeddings = embedder.encode(texts)
        
        >>> # With dimension reduction (MRL)
        >>> embedder = GeminiEmbedder(
        ...     model_name="gemini-embedding-001",
        ...     output_dimensionality=768  # Reduce from 3072
        ... )
    """
    
    # Model specifications | مواصفات النماذج
    MODEL_SPECS = {
        "gemini-embedding-001": {
            "dimensions": 3072,  # Default full dimensions
            "max_tokens": 8192,  # Increased context length
            "max_batch_size": 100,
            "output_dimensionality_support": True,
            "min_output_dimensionality": 1,
            "recommended_dimensions": [768, 1536, 3072],  # MRL recommended sizes
            "arabic_support": "excellent",
            "languages": "100+",
            "release_date": "2025",
            "status": "GA"
        },
        "text-embedding-004": {
            "dimensions": 768,
            "max_tokens": 2048,
            "max_batch_size": 100,
            "output_dimensionality_support": True,
            "min_output_dimensionality": 1,
            "arabic_support": "excellent",
            "languages": "100+",
            "release_date": "2024",
            "status": "deprecated",
            "deprecation_date": "2026-01-14"
        },
        "embedding-001": {
            "dimensions": 768,
            "max_tokens": 2048,
            "max_batch_size": 100,
            "output_dimensionality_support": False,
            "arabic_support": "excellent",
            "languages": "100+",
            "release_date": "2023",
            "status": "deprecated",
            "deprecation_date": "2025-08-14"
        }
    }
    
    # Task types with descriptions | أنواع المهام مع الوصف
    TASK_TYPES = {
        "RETRIEVAL_QUERY": {
            "ar": "استعلام البحث - للأسئلة والبحث",
            "en": "Search query - for questions and search",
            "use_case": "Use for search queries, questions"
        },
        "RETRIEVAL_DOCUMENT": {
            "ar": "وثيقة للبحث - للمستندات والمحتوى",
            "en": "Document - for documents and content",
            "use_case": "Use for documents, passages, content to be searched"
        },
        "SEMANTIC_SIMILARITY": {
            "ar": "التشابه الدلالي - لقياس التشابه",
            "en": "Semantic similarity - for measuring similarity",
            "use_case": "Use for measuring text similarity"
        },
        "CLASSIFICATION": {
            "ar": "التصنيف - لتصنيف النصوص",
            "en": "Classification - for text classification",
            "use_case": "Use for text classification tasks"
        },
        "CLUSTERING": {
            "ar": "التجميع - لتجميع النصوص المتشابهة",
            "en": "Clustering - for grouping similar texts",
            "use_case": "Use for clustering similar texts"
        }
    }
    
    def __init__(
        self,
        model_name: str = "gemini-embedding-001",
        api_key: Optional[str] = None,
        task_type: Optional[str] = None,
        output_dimensionality: Optional[int] = None,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        batch_size: int = 50,
        max_length: Optional[int] = None,
        cache_embeddings: bool = True,
        show_progress: bool = False,
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        track_usage: bool = True,
        **kwargs
    ):
        """
        Initialize Gemini Embedder with NEW google-genai SDK\n
        
        Args:
            model_name: Gemini model name
            api_key:  API key For Gemini | Gemini API key
            task_type: Task type (Retrival , Semantic Similarity, etc.)
            output_dimensionality: Output dimensions
            device: Device (not used for API)
            normalize_embeddings: Normalize embeddings
            batch_size: Batch size for processing
            max_length: Maximum text length
            cache_embeddings: Enable caching
            show_progress:  Show progress bar
            timeout:  Request timeout in seconds
            max_retries:  Maximum number of retries
            retry_delay: Base retry delay (seconds)
            track_usage:  Track usage statistics
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
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError(
                "   Google API key required\n"
                "   Provide via api_key parameter or GOOGLE_API_KEY environment variable\n"
                "   Get your key from: https://aistudio.google.com/app/apikey"
            )
        
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.track_usage = track_usage
        
        # Initialize usage stats | تهيئة إحصائيات الاستخدام
        if self.track_usage:
            self.usage_stats = GeminiUsageStats()
        
        # Store model name
        self.model_name = model_name
        
        # Get model specs | الحصول على مواصفات النموذج
        self.model_spec = self.MODEL_SPECS.get(model_name, {
            "dimensions": 768,
            "max_tokens": 2048,
            "max_batch_size": 50,
            "output_dimensionality_support": False,
            "arabic_support": "unknown",
            "languages": "unknown",
            "status": "unknown"
        })
        
        # Check if model is deprecated | التحقق من حالة النموذج
        if self.model_spec.get("status") == "deprecated":
            deprecation_date = self.model_spec.get("deprecation_date", "Unknown")
            logger.warning(
                f"⚠️ {model_name}  (deprecated)\n"
                f"   📅 Deprecation date: {deprecation_date}\n"
                f"   💡 Recommended: gemini-embedding-001\n"
                f"   Warning: Model is deprecated"
            )
        
        self.default_dimensions = self.model_spec["dimensions"]
        
        # Validate and adjust batch size | التحقق من حجم الدفعة وتعديله
        max_batch = self.model_spec.get("max_batch_size", 100)
        if self.batch_size > max_batch:
            logger.warning(
                f"   Batch size exceeds maximum {self.batch_size}, using {max_batch}"
            )
            self.batch_size = max_batch
        
        # Validate and set task type | التحقق من نوع المهمة وتحديده
        if task_type:
            if task_type not in self.TASK_TYPES:
                available_types = ', '.join(self.TASK_TYPES.keys())
                logger.warning(
                    f"   Unknown task type: {task_type}\n"
                    f"   Supported: {available_types}"
                )
            else:
                task_info = self.TASK_TYPES[task_type]
                logger.info(
                    f"🎯 Task type: {task_type}\n"
                    f"   📝 {task_info['ar']}\n"
                    f"   💡 {task_info['use_case']}"
                )
        self.task_type = task_type
        
        # Set output dimensionality | تحديد أبعاد المخرجات
        if output_dimensionality:
            if not self.model_spec.get("output_dimensionality_support", False):
                logger.warning(
                    f"   Model doesn't support dimension change\n"
                    f"   Using default: {self.default_dimensions}"
                )
                self.output_dimensionality = None
            else:
                min_dim = self.model_spec.get("min_output_dimensionality", 1)
                max_dim = self.default_dimensions
                
                if output_dimensionality < min_dim or output_dimensionality > max_dim:
                    logger.warning(
                        f"   Requested dimensions out of range\n"
                        f"   Using default: {self.default_dimensions}"
                    )
                    self.output_dimensionality = None
                else:
                    self.output_dimensionality = output_dimensionality
                    recommended = self.model_spec.get("recommended_dimensions", [])
                    if recommended and output_dimensionality not in recommended:
                        logger.info(
                            f"   Recommended dimensions for best quality: {recommended}"
                        )
                    logger.info(
                        f"📉 Dimension reduction: {self.default_dimensions} → {output_dimensionality}"
                    )
        else:
            self.output_dimensionality = None
        
        # Initialize NEW google-genai client | تهيئة عميل google-genai الجديد
        try:
            from google import genai
            from google.genai import types
            from google.genai.errors import APIError
            
            # Create client with API key
            self.client = genai.Client(api_key=self.api_key)
            self.types = types
            self.APIError = APIError
            
            logger.info("✅ New Gemini client initialized (google-genai)")
            
        except ImportError as e:
            raise ImportError(
                "   google-genai library not installed\n"
                "   ⚠️ IMPORTANT: Old library (google-generativeai) is deprecated!\n"
                "   Install new library: pip install google-genai"
            )
        
        # Validate API key with a test request | التحقق من المفتاح بطلب تجريبي
        self._validate_api_key()
        
        final_dims = self.output_dimensionality or self.default_dimensions
        
        logger.info(
            f"✅ Gemini Embedder Initialized \n"
            f"   📌 Model: {model_name}\n"
            f"   📊 Status: {self.model_spec.get('status', 'unknown')}\n"
            f"   🔢 Dimensions: {final_dims}\n"
            f"   📦 Batch size: {self.batch_size}\n"
            f"   🎯 Task type: {task_type or 'Auto'}\n"
            f"   🌍 Arabic support: {self.model_spec['arabic_support']}\n"
            f"   🌐 Languages: {self.model_spec['languages']}\n"
            f"   💾 Caching: {'active' if cache_embeddings else 'unactive'}\n"
            f"   🔄 Max retries: {max_retries}"
        )
    
    def _validate_api_key(self):
        """
        Validate API key by making a test request
        """
        try:
            logger.debug("🔐 Validating API key...")
            
            # Prepare config | تحضير الإعدادات
            config = None
            if self.task_type or self.output_dimensionality:
                config_params = {}
                if self.task_type:
                    config_params['task_type'] = self.task_type
                if self.output_dimensionality:
                    config_params['output_dimensionality'] = self.output_dimensionality
                config = self.types.EmbedContentConfig(**config_params)
            
            # Make test request
            result = self.client.models.embed_content(
                model=self.model_name,
                contents="test",
                config=config
            )
            
            logger.info("✅ API key valid")
            
        except self.APIError as e:
            error_msg = str(e)
            if "API key not valid" in error_msg or "INVALID_ARGUMENT" in error_msg:
                logger.error(f"❌ Invalid API key")
                raise ValueError(
                    f"   Get your key from:\n"
                    f"   https://aistudio.google.com/app/apikey\n"
                    f"   Invalid API key"
                )
            elif "not found" in error_msg.lower():
                logger.error(f"❌ Model not available")
                raise ValueError(
                    f"    Model available: {', '.join(self.MODEL_SPECS.keys())}\n"
                    f"   Model not available"
                )
            else:
                logger.warning(f"⚠️ Validation warning: {e}")
        
        except Exception as e:
            logger.warning(f"⚠️ Warning during validation: {e}")
    
    def _process_batch(self, texts: List[str]) -> Tuple[np.ndarray, int]:
        """
        Process a batch of texts with retry logic using NEW API
        
        Args:
            texts: List of texts
            
        Returns:
            (Array of embeddings, character count)
        """
        char_count = sum(len(text) for text in texts)
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # Prepare config | تحضير الإعدادات
                config = None
                if self.task_type or self.output_dimensionality:
                    config_params = {}
                    if self.task_type:
                        config_params['task_type'] = self.task_type
                    if self.output_dimensionality:
                        config_params['output_dimensionality'] = self.output_dimensionality
                    config = self.types.EmbedContentConfig(**config_params)
                
                # Make API call with NEW SDK
                # استدعاء API الجديد
                result = self.client.models.embed_content(
                    model=self.model_name,
                    contents=texts,
                    config=config
                )
                
                # Extract embeddings from NEW response format
                # استخراج التضمينات من تنسيق الاستجابة الجديد
                # result.embeddings is a list of ContentEmbedding objects
                # Each ContentEmbedding has a 'values' attribute
                embeddings_list = []
                for emb_obj in result.embeddings:
                    if hasattr(emb_obj, 'values'):
                        embeddings_list.append(emb_obj.values)
                    else:
                        # Fallback if structure is different
                        embeddings_list.append(emb_obj)
                
                embeddings = np.array(embeddings_list, dtype=np.float32)
                
                # Ensure 2D array shape
                if embeddings.ndim == 1:
                    embeddings = embeddings.reshape(1, -1)
                
                # Validate number of embeddings matches input
                if embeddings.shape[0] != len(texts):
                    logger.error(
                        f"❌ Embedding count mismatch:\n"
                        f"   Expected: {len(texts)}\n"
                        f"   Received: {embeddings.shape[0]}"
                    )
                    raise ValueError(f"Embedding count mismatch: expected {len(texts)}, got {embeddings.shape[0]}")
                
                # Validate embedding dimensions
                expected_dim = self.output_dimensionality or self.default_dimensions
                if embeddings.shape[1] != expected_dim:
                    logger.warning(
                        f"⚠️  Embedding dimension mismatch:\n"
                        f"   Expected: {expected_dim}\n"
                        f"   Received:({embeddings.shape[1]}) not match with expected ({expected_dim})"
                    )
                
                # Track usage
                if self.track_usage:
                    self.usage_stats.add_request(len(texts), char_count)
                
                return embeddings, char_count
                
            except self.APIError as e:
                error_msg = str(e)
                
                # Check for rate limiting
                if "429" in error_msg or "quota" in error_msg.lower():
                    last_error = f"📊 Resource exhausted / Rate limited"
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"{last_error} (retry {attempt + 1}/{self.max_retries})\n"
                        f"   ⏳ waiting {wait_time:.1f}s"
                    )
                    if self.track_usage:
                        self.usage_stats.add_error()
                    if attempt < self.max_retries - 1:
                        time.sleep(wait_time)
                
                # Check for timeout
                elif "timeout" in error_msg.lower() or "deadline" in error_msg.lower():
                    last_error = f"⏱️ Request timeout"
                    logger.warning(f"{last_error} (retry {attempt + 1}/{self.max_retries})")
                    if self.track_usage:
                        self.usage_stats.add_error()
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (2 ** attempt))
                
                else:
                    # Other API errors - don't retry
                    last_error = f"❌ API error: {error_msg[:100]}"
                    logger.error(f"{last_error}")
                    if self.track_usage:
                        self.usage_stats.add_error()
                    raise
            
            except Exception as e:
                last_error = f"❌ Unexpected error: {str(e)[:100]}"
                logger.error(f"{last_error}")
                if self.track_usage:
                    self.usage_stats.add_error()
                raise
        
        # All retries failed
        error_msg = f"❌ all retries failed after  {self.max_retries} retries\n    last error: {last_error}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    def encode(
        self,
        texts: Union[str, List[str]],
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        task_type: Optional[str] = None,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Convert texts to embeddings with optimized processing
        
        Args:
            texts:  Single text or list of texts
            show_progress_bar:  Show progress bar
            convert_to_numpy:  Convert to numpy array
            task_type: Task type (overrides default)
            batch_size:  Batch size override
            **kwargs:  Additional parameters
            
        Returns:
               Array of embeddings
        """
        # Override task type if provided
        original_task_type = self.task_type
        if task_type:
            if task_type not in self.TASK_TYPES:
                logger.warning(f"⚠️ task type not supported: {task_type}")
            self.task_type = task_type
        
        try:
            # Convert single text to list
            if isinstance(texts, str):
                texts = [texts]
            
            # Validate input
            if not texts:
                raise ValueError("❌ Empty text list provided")
            
            # Filter empty texts
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
            total_chars = 0
            
            # Use provided batch_size or default
            effective_batch_size = batch_size or self.batch_size
            
            # Progress tracking
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
                    logger.warning("⚠️ tqdm not installed. Install: pip install tqdm")
            else:
                batch_iterator = range(0, total_texts, effective_batch_size)
            
            # Process in batches
            cache_hits = 0
            cache_misses = 0
            
            for i in batch_iterator:
                batch = texts[i:i + effective_batch_size]
                batch_texts_to_process = []
                batch_indices = []
                cached_embeddings = {}
                
                # Check cache for each text
                cache_key_suffix = f"_{self.task_type}" if self.task_type else ""
                cache_key_suffix += f"_dim{self.output_dimensionality}" if self.output_dimensionality else ""
                
                for j, text in enumerate(batch):
                    cache_key = f"{text}{cache_key_suffix}"
                    if self.cache_embeddings and cache_key in self._cache:
                        cached_embeddings[j] = self._cache[cache_key]
                        cache_hits += 1
                        self._stats['cache_hits'] += 1
                        if self.track_usage:
                            self.usage_stats.cache_hits += 1
                    else:
                        batch_texts_to_process.append(text)
                        batch_indices.append(j)
                        cache_misses += 1
                
                # Process non-cached texts
                if batch_texts_to_process:
                    try:
                        batch_embeddings, chars_processed = self._process_batch(batch_texts_to_process)
                        total_chars += chars_processed
                        
                        # Cache new embeddings
                        if self.cache_embeddings:
                            for text, embedding in zip(batch_texts_to_process, batch_embeddings):
                                cache_key = f"{text}{cache_key_suffix}"
                                self._cache[cache_key] = embedding
                        
                        # Combine cached and new embeddings
                        final_batch_embeddings = np.zeros(
                            (len(batch), batch_embeddings.shape[1]),
                            dtype=np.float32
                        )
                        
                        # Fill in cached embeddings
                        for idx, emb in cached_embeddings.items():
                            final_batch_embeddings[idx] = emb
                        
                        # Fill in new embeddings
                        for idx, emb_idx in enumerate(batch_indices):
                            final_batch_embeddings[emb_idx] = batch_embeddings[idx]
                            
                    except Exception as e:
                        logger.error(f"❌ failed to process batch {i//effective_batch_size + 1}: {e}")
                        raise
                else:
                    # All from cache
                    final_batch_embeddings = np.array(
                        [cached_embeddings[j] for j in range(len(batch))],
                        dtype=np.float32
                    )
                
                all_embeddings.append(final_batch_embeddings)
            
            # Combine all batches
            embeddings_array = np.vstack(all_embeddings)
            
            # Normalize if requested
            if self.normalize_embeddings:
                norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1, norms)
                embeddings_array = embeddings_array / norms
            
            # Update statistics
            elapsed_time = time.time() - start_time
            self._stats['total_encodings'] += 1
            self._stats['total_texts'] += total_texts
            self._stats['total_time'] += elapsed_time
            self._stats['avg_time_per_text'] = self._stats['total_time'] / self._stats['total_texts']
            
            # Calculate cache efficiency
            cache_hit_rate = (cache_hits / total_texts * 100) if total_texts > 0 else 0
            
            logger.info(
                f"✅ Generated: {total_texts}\n"
                f"   📊 Avg time: {elapsed_time/total_texts:.3f}s/text\n"
                f"   💾 Cache: {cache_hits}/{total_texts} ({cache_hit_rate:.1f}%)\n"
                f"   📝 Total chars: {total_chars:,}\n"
                f"   📏 Dimension: {self.embedding_dim}"
            )
            
            return embeddings_array
            
        finally:
            # Restore original task type
            self.task_type = original_task_type
    
    @property
    def embedding_dim(self) -> int:
        """
        Get embedding dimension
        
        Returns:
            Embedding vector dimension
        """
        return self.output_dimensionality or self.default_dimensions
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get detailed usage statistics
        
        Returns:
           Usage statistics dictionary
        """
        stats = self._stats.copy()
        stats['embedding_dim'] = self.embedding_dim
        stats['model_name'] = self.model_name
        stats['model_status'] = self.model_spec.get('status', 'unknown')
        stats['task_type'] = self.task_type
        stats['cache_enabled'] = self.cache_embeddings
        stats['cache_size'] = len(self._cache) if self.cache_embeddings else 0
        
        # Calculate cache hit rate
        if stats['total_texts'] > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['total_texts']
        else:
            stats['cache_hit_rate'] = 0.0
        
        # Add API usage stats if tracking
        if self.track_usage:
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
            'configured_dimensions': self.embedding_dim,
            'task_type': self.task_type,
            'normalize_embeddings': self.normalize_embeddings,
            'batch_size': self.batch_size,
            'cache_enabled': self.cache_embeddings,
            'usage_tracking_enabled': self.track_usage,
            'sdk_version': 'google-genai (NEW)'
        })
        
        if self.track_usage:
            info['usage_stats'] = self.usage_stats.get_summary()
        
        return info
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with summary"""
        if self.track_usage:
            summary = self.usage_stats.get_summary()
            logger.info(
                f"\n📊 Session Summary:\n"
                f"   📝 Total requests: {summary['total_requests']}\n"
                f"   🔤 Total texts: {summary['total_texts']}\n"
                f"   📝 Total chars: {summary['total_characters']:,}\n"
                f"   ⏱️ Elapsed: {summary['elapsed_time_seconds']:.1f}s\n"
                f"   ✅ Success rate: {summary['success_rate']:.1%}"
            )

