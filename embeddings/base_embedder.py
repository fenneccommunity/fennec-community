"""
Base Embedder Interface - Enhanced Version
"""

from abc import ABC, abstractmethod
from typing import List, Union, Optional, Dict, Any
import numpy as np
import logging
from contextlib import contextmanager
import time
from .config_embedder import EmbedderConfig

config=EmbedderConfig()
logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """
    Base interface for all text embedders
    """
    
    def __init__(self, 
                 model_name: str,
                 device: Optional[str] = None,
                 normalize_embeddings: bool = config.normalize_embeddings,
                 batch_size: int = config.batch_size,
                 max_length: Optional[int] = config.max_seq_length,
                 cache_embeddings: bool = config.enable_cache,
                 show_progress: bool = config.show_progress_bar,
                 **kwargs):
        """
        Args:
            model_name:  Model name
            device:  Device to use (cuda/cpu/mps/auto)
            normalize_embeddings: Normalize embeddings
            batch_size:  Batch size
            max_length:  Max text length
            cache_embeddings:  Enable caching
            show_progress: Show progress
            **kwargs:  Additional parameters
        """
        self.model_name = model_name
        self.device = device or self._get_best_device()
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.max_length = max_length
        self.cache_embeddings = cache_embeddings
        self.show_progress = show_progress
        self.kwargs = kwargs
        
        # إحصائيات الأداء | Performance statistics
        # Performance statistics | إحصائيات الأداء
        self._stats = {
            'total_encodings': 0,      # إجمالي عمليات التحويل | Total encodings
            'total_texts': 0,          # إجمالي النصوص | Total texts
            'cache_hits': 0,           # عدد مرات استخدام الذاكرة المؤقتة | Cache hits count
            'total_time': 0.0,         # الوقت الإجمالي | Total time
            'avg_time_per_text': 0.0   # متوسط الوقت لكل نص | Average time per text
        }
        
        # التخزين المؤقت | Cache
        # Cache | التخزين المؤقت
        if cache_embeddings:
            self._cache = {}
            logger.info("✅ Caching enabled")
        
        logger.info(f"🚀 Intialized {self.__class__.__name__}")
        logger.info(f"📌 Model: {model_name}")
        logger.info(f"🖥️  Device: {self.device}")
        logger.info(f"📦 Batch size: {batch_size}")
    
    @abstractmethod
    def encode(self, 
               texts: Union[str, List[str]], 
               show_progress_bar: bool = False,
               convert_to_numpy: bool = True,
               **kwargs) -> np.ndarray:
        """
        Convert texts to embeddings
        
        Args:
            texts: Single text or list of texts
            show_progress_bar:  Show progress bar
            convert_to_numpy:  Convert to numpy array
            **kwargs:  Additional parameters
        
        Returns:
            Text embeddings
        """
        pass
    
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """
        Get embedding dimension
        """
        pass
    
    def encode_with_cache(self, 
                          texts: Union[str, List[str]], 
                          **kwargs) -> np.ndarray:
        """
        Encode texts with caching support
        
        Args:
            texts:  Texts
            **kwargs:  Additional parameters
        
        Returns:
            Embeddings
        """
        if not self.cache_embeddings:
            return self.encode(texts, **kwargs)
        
        # التعامل مع نص واحد | Handle single text
        # Handle single text | التعامل مع نص واحد
        if isinstance(texts, str):
            cache_key = self._get_cache_key(texts)
            if cache_key in self._cache:
                if isinstance(self._stats, dict): self._stats['cache_hits'] += 1
                else: self._stats.cache_hits += 1
                return self._cache[cache_key]
            
            embedding = self.encode(texts, **kwargs)
            self._cache[cache_key] = embedding
            return embedding
        
        # التعامل مع قائمة نصوص | Handle list of texts
        # Handle list of texts | التعامل مع قائمة نصوص
        embeddings = []
        texts_to_encode = []
        text_indices = []
        
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if cache_key in self._cache:
                embeddings.append(self._cache[cache_key])
                if isinstance(self._stats, dict): self._stats['cache_hits'] += 1
                else: self._stats.cache_hits += 1
            else:
                texts_to_encode.append(text)
                text_indices.append(i)
        
        # تحويل النصوص الجديدة | Encode new texts
        # Encode new texts | تحويل النصوص الجديدة
        if texts_to_encode:
            new_embeddings = self.encode(texts_to_encode, **kwargs)
            
            # إضافة للتخزين المؤقت | Add to cache
            # Add to cache | إضافة للتخزين المؤقت
            for text, emb in zip(texts_to_encode, new_embeddings):
                cache_key = self._get_cache_key(text)
                self._cache[cache_key] = emb
            
            # دمج النتائج | Merge results
            # Merge results | دمج النتائج
            all_embeddings = np.zeros((len(texts), new_embeddings.shape[1]))
            
            # إضافة المتجهات من الذاكرة المؤقتة | Add embeddings from cache
            # Add embeddings from cache | إضافة المتجهات من الذاكرة المؤقتة
            cache_idx = 0
            encode_idx = 0
            for i in range(len(texts)):
                if i in text_indices:
                    all_embeddings[i] = new_embeddings[encode_idx]
                    encode_idx += 1
                else:
                    all_embeddings[i] = embeddings[cache_idx]
                    cache_idx += 1
            
            return all_embeddings
        
        return np.array(embeddings)
    
    def _get_cache_key(self, text: str) -> str:
        """
        Generate cache key
        """
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()
    
    def _get_best_device(self) -> str:
        """
        Automatically select best available device
        """
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                logger.info(f"🎮 CUDA available: {device_name}")
                return "cuda"
            elif torch.backends.mps.is_available():
                logger.info("🍎 MPS available (Apple Silicon)")
                return "mps"
            else:
                logger.info("💻 Using CPU")
                return "cpu"
        except ImportError:
            logger.warning("⚠️ PyTorch not available, using CPU")
            return "cpu"
    
    def similarity(self, 
                   text1: Union[str, np.ndarray], 
                   text2: Union[str, np.ndarray],
                   metric: str = 'cosine') -> float:
        """
        Calculate similarity between two texts
        
        Args:
            text1:  Text or embedding
            text2:  Text or embedding
            metric: Metric type ('cosine', 'dot', 'euclidean')
        
        Returns:
            Similarity score
        """
        # الحصول على المتجهات | Get embeddings
        # Get embeddings | الحصول على المتجهات
        emb1 = self.encode(text1) if isinstance(text1, str) else text1
        emb2 = self.encode(text2) if isinstance(text2, str) else text2
        
        # ضمان شكل المتجهات | Ensure correct shape
        # Ensure correct shape | ضمان شكل المتجهات
        if emb1.ndim == 2:
            emb1 = emb1[0]
        if emb2.ndim == 2:
            emb2 = emb2[0]
        
        # حساب التشابه حسب المقياس | Calculate similarity based on metric
        # Calculate similarity based on metric | حساب التشابه حسب المقياس
        if metric == 'cosine' or metric == 'dot':
            similarity = np.dot(emb1, emb2)
        elif metric == 'euclidean':
            similarity = 1.0 / (1.0 + np.linalg.norm(emb1 - emb2))
        else:
            raise ValueError(f"Unsupported metric: {metric}  , Supported Metrics [ cosine , dot , euclidean ]")
        
        return float(similarity)
    
    def batch_similarity(self, 
                        query: Union[str, np.ndarray], 
                        texts: List[str],
                        top_k: Optional[int] = None) -> Union[np.ndarray, List[tuple]]:
        """
        Calculate similarity between query and list of texts
        
        Args:
            query:  Query from user
            texts:  List of texts
            top_k:  Return top k results
        
        Returns:
            Array of similarity scores or list of (index, score) tuples
        """
        start_time = time.time()
        
        # الحصول على المتجهات | Get embeddings
        # Get embeddings | الحصول على المتجهات
        query_emb = self.encode(query) if isinstance(query, str) else query
        text_embs = self.encode(texts) if isinstance(texts[0], str) else np.array(texts)
        
        # ضمان الشكل الصحيح | Ensure correct shape
        # Ensure correct shape | ضمان الشكل الصحيح
        if query_emb.ndim == 2:
            query_emb = query_emb[0]
        
        # حساب التشابه | Calculate similarities
        # Calculate similarities | حساب التشابه
        similarities = np.dot(text_embs, query_emb)
        
        # تحديث الإحصائيات | Update statistics
        # Update statistics | تحديث الإحصائيات
        elapsed = time.time() - start_time
        if isinstance(self._stats, dict):
            self._stats['total_time'] += elapsed
        else:
            self._stats.total_time += elapsed
        
        # إرجاع أفضل k نتيجة إذا طُلب | Return top k if requested
        # Return top k if requested | إرجاع أفضل k نتيجة إذا طُلب
        if top_k is not None:
            top_indices = np.argsort(similarities)[::-1][:top_k]
            return [(int(idx), float(similarities[idx])) for idx in top_indices]
        
        return similarities
    
    def validate_connection(self, 
                          test_text: str = "مرحباً Hello",
                          detailed: bool = False) -> Dict[str, Any]:
        """
        Validate that the embedder works correctly
        
        Args:
            test_text:  Test text
            detailed:  Return detailed info
        
        Returns:
            Results dictionary
        """
        try:
            start_time = time.time()
            embedding = self.encode(test_text)
            encoding_time = time.time() - start_time
            
            if embedding is not None and len(embedding) > 0:
                result = {
                    "success": True,
                    "reason": "✅ Embedder works successfully",
                    "embedding_dim": self.embedding_dim,
                    "encoding_time": f"{encoding_time:.3f}s"
                }
                
                if detailed:
                    result.update({
                        "model_name": self.model_name,
                        "device": self.device,
                        "test_text_length": len(test_text),
                        "embedding_shape": embedding.shape,
                        "embedding_norm": float(np.linalg.norm(embedding)),
                        "embedding_mean": float(np.mean(embedding)),
                        "embedding_std": float(np.std(embedding))
                    })
                
                return result
            else:
                return {
                    "success": False,
                    "reason": "❌ Empty embedding",
                    "embedding_dim": 0
                }
        
        except Exception as e:
            logger.error(f"❌ Validation error: {str(e)}")
            return {
                "success": False,
                "reason": f"❌ Error: {str(e)}",
                "embedding_dim": 0,
                "error_type": type(e).__name__
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Detailed model information
        """
        return {
            'model_name': self.model_name,
            'model_class': self.__class__.__name__,
            'embedding_dim': self.embedding_dim,
            'device': self.device,
            'normalize_embeddings': self.normalize_embeddings,
            'batch_size': self.batch_size,
            'max_length': self.max_length,
            'cache_enabled': self.cache_embeddings,
            'additional_kwargs': self.kwargs
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Performance statistics
        """
        # دعم كل من dict و dataclass | Support both dict and dataclass _stats
        if isinstance(self._stats, dict):
            if self._stats['total_texts'] > 0:
                self._stats['avg_time_per_text'] = (
                    self._stats['total_time'] / self._stats['total_texts']
                )
            stats = self._stats.copy()
            if self.cache_embeddings and self._stats['total_encodings'] > 0:
                stats['cache_hit_rate'] = (
                    self._stats['cache_hits'] / self._stats['total_encodings']
                )
                if hasattr(self, '_cache'):
                    stats['cache_size'] = len(self._cache)
        else:
            # EmbeddingStats dataclass (e.g. HuggingFaceEmbedder)
            stats = self._stats.to_dict() if hasattr(self._stats, 'to_dict') else vars(self._stats).copy()
            if self.cache_embeddings and hasattr(self, '_cache'):
                stats['cache_size'] = len(self._cache)

        return stats
    
    def reset_stats(self):
        """
        Reset statistics
        """
        self._stats = {
            'total_encodings': 0,
            'total_texts': 0,
            'cache_hits': 0,
            'total_time': 0.0,
            'avg_time_per_text': 0.0
        }
        logger.info("🔄 Statistics reset")
    
    def clear_cache(self):
        """
        Clear cache
        """
        if self.cache_embeddings:
            cache_size = len(self._cache)
            self._cache.clear()
            logger.info(f"🧹 Cleared {cache_size} items from cache")
    
    @contextmanager
    def timing(self, operation: str = "encoding"):
        """
        Context manager for timing
        
        Usage:
            with embedder.timing("batch_encoding"):
                embeddings = embedder.encode(texts)
        """
        start = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start
            logger.info(f"⏱️  {operation}: {elapsed:.3f}s")
    
    def cleanup(self):
        """
        Clean up resources
        """
        try:
            # مسح التخزين المؤقت | Clear cache
            # Clear cache | مسح التخزين المؤقت
            if self.cache_embeddings:
                self.clear_cache()
            
            # تنظيف ذاكرة GPU | Clean GPU memory
            # Clean GPU memory | تنظيف ذاكرة GPU
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("🧹 GPU memory cleaned")
            
        except ImportError:
            pass
        
        logger.info(f"🧹 Cleaned up {self.__class__.__name__} resources")
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model={self.model_name}, "
            f"dim={self.embedding_dim}, "
            f"device={self.device}, "
            f"batch={self.batch_size})"
        )

    # ── Async API ────────────────────────────────────────────────────

    async def aencode(self, texts, show_progress_bar: bool = False,
                      convert_to_numpy: bool = True, **kwargs):
        """
        Async encoding — runs CPU-bound encode() in a thread pool.\n
        """
        import asyncio
        return await asyncio.to_thread(
            self.encode, texts, show_progress_bar, convert_to_numpy, **kwargs
        )

    async def aencode_with_cache(self, texts, **kwargs):
        """
        Async encoding with cache support.
        """
        import asyncio
        return await asyncio.to_thread(self.encode_with_cache, texts, **kwargs)

    async def abatch_similarity(self, query, candidates, **kwargs):
        """
        Async batch similarity computation.
        """
        import asyncio
        return await asyncio.to_thread(self.batch_similarity, query, candidates, **kwargs)

    # ── Async context manager ─────────────────────────────────────────
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False

    def __del__(self):
        """ Automatic cleanup"""
        try:
            self.cleanup()
        except:
            pass