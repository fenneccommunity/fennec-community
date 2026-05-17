from typing import List, Optional, Union, Dict, Tuple, Any
import numpy as np
import logging
import re
import time
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from .base_embedder import BaseEmbedder
from .config_embedder import EmbedderConfig
config = EmbedderConfig()
# إعداد نظام السجلات / Setup logging
logger = logging.getLogger(__name__)


@dataclass
class ArabicProcessingStats:
    """Arabic text processing statistics"""
    total_texts: int = 0
    normalized_texts: int = 0
    avg_normalization_time: float = 0.0
    removed_diacritics: int = 0
    normalized_hamzas: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get statistics summary"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        return {
            'total_texts': self.total_texts,
            'normalized_texts': self.normalized_texts,
            'normalization_rate': self.normalized_texts / max(1, self.total_texts),
            'avg_normalization_time_ms': self.avg_normalization_time * 1000,
            'removed_diacritics': self.removed_diacritics,
            'normalized_hamzas': self.normalized_hamzas,
            'elapsed_time_seconds': round(elapsed, 2)
        }


class ArabicEmbedder(BaseEmbedder):
    """
    Enhanced Arabic text embedder with multi-model support
    
    Supported Models:
    - multilingual: paraphrase-multilingual-mpnet-base-v2 (Best overall)
    - labse: sentence-transformers/LaBSE (Good for cross-lingual)
    - arabert: aubmindlab/bert-base-arabertv2 (Arabic-specific)
    - camelbert: CAMeL-Lab/bert-base-arabic-camelbert-msa (Modern Standard Arabic)
    - arabic-mini: paraphrase-multilingual-MiniLM-L12-v2 (Lightweight)
    
    Features:
    - Advanced Arabic normalization 
    - Diacritic removal 
    - Hamza normalization 
    - Processing statistics 
    - Model download management 
    """
    
    # النماذج الموصى بها / Recommended models
    RECOMMENDED_MODELS = {
        'multilingual': {
            'name': 'paraphrase-multilingual-mpnet-base-v2',
            'dim': 768,
            'description_ar': 'أفضل نموذج متعدد اللغات - دقة عالية',
            'description_en': 'Best multilingual model - high accuracy',
            'best_for': 'General purpose, high quality'
        },
        'multilingual-mini': {
            'name': 'paraphrase-multilingual-MiniLM-L12-v2',
            'dim': 384,
            'description_ar': 'نموذج خفيف وسريع - توازن جيد',
            'description_en': 'Lightweight and fast - good balance',
            'best_for': 'Speed and efficiency'
        },
        'labse': {
            'name': 'sentence-transformers/LaBSE',
            'dim': 768,
            'description_ar': 'ممتاز للبحث متعدد اللغات',
            'description_en': 'Excellent for cross-lingual search',
            'best_for': 'Cross-lingual retrieval'
        },
        'arabert': {
            'name': 'aubmindlab/bert-base-arabertv2',
            'dim': 768,
            'description_ar': 'متخصص للعربية - دقة عالية',
            'description_en': 'Arabic-specific - high accuracy',
            'best_for': 'Arabic-only tasks'
        },
        'camelbert': {
            'name': 'CAMeL-Lab/bert-base-arabic-camelbert-msa',
            'dim': 768,
            'description_ar': 'مصمم للعربية الفصحى',
            'description_en': 'Designed for Modern Standard Arabic',
            'best_for': 'Formal Arabic text'
        }
    }
    
    # نماذج التطبيع / Normalization patterns
    ARABIC_NORMALIZATION_PATTERNS = {
        'hamza_on_alef': re.compile('[أإآ]'),
        'hamza_on_waw': re.compile('ؤ'),
        'hamza_on_ya': re.compile('ئ'),
        'alef_maksura': re.compile('ى'),
        'taa_marbuta': re.compile('ة'),
        'diacritics': re.compile('[\u064B-\u065F]'),  # All diacritics
        'tatweel': re.compile('ـ'),
        'repeated_chars': re.compile(r'(.)\1{2,}'),  # 3+ repeated chars
    }
    
    def __init__(self, 
                 model_name: Optional[str] = None,
                 device: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 normalize_embeddings: bool = config.normalize_embeddings,
                 batch_size: int = config.batch_size,
                 enable_preprocessing: bool = config.enable_preprocessing,
                 max_seq_length: int = config.max_seq_length,
                 normalization_level: str = 'standard',
                 track_processing_stats: bool = config.track_processing_stats,
                 auto_download: bool = config.auto_download,
                 trust_remote_code: bool = config.trust_remote_code,
                 **kwargs):
        """
        Initialize the Arabic text embedder
        
        Args:
            model_name:
                       Model name or key from RECOMMENDED_MODELS
            device: 
                   Device to use (cuda/cpu/mps/auto)
            cache_dir: 
                      Directory for caching models
            normalize_embeddings: 
                                 Normalize embeddings (recommended)
            batch_size: 
                       Batch size for processing
            enable_preprocessing: 
                                 Enable Arabic preprocessing
            max_seq_length: 
                           Maximum sequence length
            normalization_level: 
                                Normalization level
            track_processing_stats: 
                                   Track processing statistics
            auto_download: 
                          Auto-download model if not present
            trust_remote_code: 
                              Trust remote code (for certain models)
            **kwargs:  Additional parameters
        """
        # اختيار النموذج / Select model
        if model_name is None:
            model_name = 'multilingual-mini'
            logger.info("ℹ️ Using default model: multilingual-mini")
        
        # التحقق من المفتاح / Check if it's a key
        if model_name in self.RECOMMENDED_MODELS:
            model_info = self.RECOMMENDED_MODELS[model_name]
            actual_model = model_info['name']
            logger.info(
                f"📦 Using recommended model:\n"
                f"   Name: {model_name}\n"
                f"   Model: {actual_model}\n"
                f"   Description: {model_info['description_ar']}\n"
                f"   Best for: {model_info['best_for']}"
            )
            self.model_key = model_name
            self.model_info = model_info
            model_name = actual_model
        else:
            self.model_key = None
            self.model_info = {'name': model_name, 'dim': 'unknown'}
        
        # استدعاء الصنف الأب / Call parent constructor
        super().__init__(model_name, device, normalize_embeddings, batch_size, **kwargs)
        
        self.cache_dir = cache_dir
        self.enable_preprocessing = enable_preprocessing
        self.max_seq_length = max_seq_length
        self.normalization_level = normalization_level
        self.track_processing_stats = track_processing_stats
        self.auto_download = auto_download
        self.trust_remote_code = trust_remote_code
        
        # إحصائيات المعالجة / Processing statistics
        if self.track_processing_stats:
            self.processing_stats = ArabicProcessingStats()
        
        # التحقق من مستوى التطبيع / Validate normalization level
        valid_levels = ['minimal', 'standard', 'aggressive']
        if normalization_level not in valid_levels:
            logger.warning(
                f"⚠️ Normalization Level Not Found: {normalization_level}\n"
                f"    availabel level : {', '.join(valid_levels)}\n"
                f"    standard will using"
            )
            self.normalization_level = 'standard'
        
        # تحميل النموذج / Load model
        self._load_model()
        
        logger.info(
            f"✅ Arabic Embedder initialized\n"
            f"   Model: {self.model_name.split('/')[-1]}\n"
            f"   Dimension: {self.embedding_dim}\n"
            f"   Device: {self.device}\n"
            f"   Max length: {self.max_seq_length}\n"
            f"   Preprocessing: {'activated' if enable_preprocessing else 'unactivate'}\n"
            f"   Normalization: {normalization_level}"
        )
    
    def _load_model(self):
        """
        Load model with error handling
        """
        try:
            from sentence_transformers import SentenceTransformer
            
            # إعداد معاملات النموذج / Setup model parameters
            model_kwargs = {}
            if self.cache_dir:
                model_kwargs['cache_folder'] = self.cache_dir
            if self.trust_remote_code:
                model_kwargs['trust_remote_code'] = True
            
            # تحميل النموذج / Load the model
            logger.info(f"📥 Loading model: {self.model_name}")
            
            try:
                self.model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    **model_kwargs
                )
            except Exception as e:
                if self.auto_download:
                    logger.info(
                        f"   Model not found locally, downloading...\n"
                        f"   This may take a while"
                    )
                    self.model = SentenceTransformer(
                        self.model_name,
                        device=self.device,
                        **model_kwargs
                    )
                else:
                    raise   Exception.add_note("the model is not available locally and auto_download is disabled ") from e
            
            # تحسينات للأداء / Performance optimizations
            self.model.max_seq_length = self.max_seq_length
            
            logger.info(
                f"✅ Model loaded successfully\n"
                f"   Embedding dimension: {self.embedding_dim}\n"
                f"   Device: {self.device}"
            )
            
        except ImportError:
            error_msg = (
                "   sentence-transformers is not installed\n"
                "   Install with: pip install sentence-transformers"
            )
            logger.error(error_msg)
            raise ImportError(error_msg)
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise
    
    def encode(self, 
               texts: Union[str, List[str]], 
               show_progress_bar: bool = config.show_progress_bar,
               convert_to_numpy: bool = config.convert_to_numpy,
               batch_size: Optional[int] = config.batch_size,
               return_valid_indices: bool = config.return_valid_indices,
               skip_preprocessing: bool = config.skip_preprocessing,
               **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, List[int]]]:
        """
        Convert texts to embeddings
        
        Args:
            texts: 
                  Single text or list of texts
            show_progress_bar: 
                             Show progress bar
            convert_to_numpy:
                            Convert to numpy array
            batch_size: 
                       Batch size (optional)
            return_valid_indices: 
                                 Return valid text indices
            skip_preprocessing: 
                               Skip preprocessing
            **kwargs:  Additional parameters
        
        Returns:
           
            Text embeddings or (embeddings, valid_indices)
        """
        # التحقق من وجود نصوص / Check for empty input
        if not texts:
            logger.warning("⚠️ No texts provided")
            empty_result = np.array([])
            return (empty_result, []) if return_valid_indices else empty_result
        
        # التحقق من نوع البيانات / Validate input type
        if isinstance(texts, str):
            single_text = True
            texts = [texts]
        elif isinstance(texts, (list, tuple)):
            single_text = False
            # التحقق من أن جميع العناصر نصوص
            # Verify all elements are strings
            if not all(isinstance(t, (str, type(None))) for t in texts):
                raise TypeError(
                    "❌ All elements in texts must be strings"
                )
        else:
            raise TypeError(
                f"   texts must be str or list, got {type(texts)}"
            )
        
        start_time = time.time()
        
        # فلترة النصوص الفارغة / Filter empty texts
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text and isinstance(text, str) and text.strip():
                valid_texts.append(text.strip())
                valid_indices.append(i)
            else:
                logger.debug(
                    f"   Skipping empty/invalid text at position {i}"
                )
        
        # التحقق من وجود نصوص صالحة / Check for valid texts
        if not valid_texts:
            logger.warning("⚠️All texts are empty/invalid")
            empty_result = np.array([])
            return (empty_result, []) if return_valid_indices else empty_result
        
        try:
            # المعالجة المسبقة للنصوص العربية / Preprocess Arabic texts
            if self.enable_preprocessing and not skip_preprocessing:
                processed_texts = self._preprocess_texts(valid_texts)
                if self.track_processing_stats:
                    self.processing_stats.total_texts += len(valid_texts)
                    self.processing_stats.normalized_texts += len(processed_texts)
                logger.debug(
                    f"✅ Preprocessed {len(processed_texts)} texts"
                )
            else:
                processed_texts = valid_texts
            
            # التضمين / Encode texts
            batch_size = batch_size or self.batch_size
            
            try:
                embeddings = self.model.encode(
                    processed_texts,
                    batch_size=batch_size,
                    show_progress_bar=show_progress_bar,
                    normalize_embeddings=self.normalize_embeddings,
                    convert_to_numpy=convert_to_numpy,
                    **kwargs
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(
                        f"   Out of memory, retrying with smaller batch size"
                    )
                    # Try with smaller batch size
                    smaller_batch = max(1, batch_size // 2)
                    embeddings = self.model.encode(
                        processed_texts,
                        batch_size=smaller_batch,
                        show_progress_bar=show_progress_bar,
                        normalize_embeddings=self.normalize_embeddings,
                        convert_to_numpy=convert_to_numpy,
                        **kwargs
                    )
                else:
                    raise
            
            elapsed_time = time.time() - start_time
            
            # تحديث الإحصائيات / Update statistics
            if self.track_processing_stats and self.enable_preprocessing:
                norm_time = elapsed_time / len(valid_texts)
                self.processing_stats.avg_normalization_time = (
                    (self.processing_stats.avg_normalization_time * 
                     (self.processing_stats.normalized_texts - len(valid_texts)) +
                     norm_time * len(valid_texts)) /
                    self.processing_stats.normalized_texts
                )
            
            logger.info(
                f"   Generated {len(embeddings)} embeddings\n"
                f"   Avg time: {elapsed_time/len(valid_texts):.3f}s/text"
            )
            
            # إرجاع النتيجة / Return result
            if single_text:
                result = embeddings[0] if len(embeddings) > 0 else np.array([])
                return (result, valid_indices) if return_valid_indices else result
            
            # إعادة بناء المصفوفة الكاملة مع القيم الفارغة
            # Rebuild full array with empty values for invalid texts
            if len(valid_indices) < len(texts):
                full_embeddings = np.zeros((len(texts), self.embedding_dim))
                for i, idx in enumerate(valid_indices):
                    full_embeddings[idx] = embeddings[i]
                embeddings = full_embeddings
            
            return (embeddings, valid_indices) if return_valid_indices else embeddings
            
        except Exception as e:
            logger.error(f"❌ Encoding error: {e}")
            raise
    
    def _preprocess_texts(self, texts: List[str]) -> List[str]:
        """
        Preprocess Arabic texts
        
        Args:
            texts: قائمة النصوص / List of texts
            
        Returns:
            نصوص معالجة / Processed texts
        """
        processed = []
        for text in texts:
            # تطبيع النص العربي / Normalize Arabic text
            text = self._normalize_arabic(text)
            
            # إزالة الأحرف الخاصة المكررة / Remove repeated special characters
            if self.normalization_level in ['standard', 'aggressive']:
                text = re.sub(r'([!?؟.])\1+', r'\1', text)
            
            # إزالة الأحرف المكررة بشكل عام (فقط في aggressive)
            # Remove repeated characters in general (only in aggressive)
            if self.normalization_level == 'aggressive':
                text = self.ARABIC_NORMALIZATION_PATTERNS['repeated_chars'].sub(r'\1\1', text)
            
            # إزالة أي مسافات زائدة / Remove any remaining extra spaces
            text = ' '.join(text.split())
            
            processed.append(text)
        
        return processed
    
    def _normalize_arabic(self, text: str) -> str:
        """
        Normalize Arabic text
        
        Normalizations applied based on level | التطبيعات المطبقة حسب المستوى:
        
        Minimal:
        - Remove diacritics only | إزالة التشكيل فقط
        
        Standard (default):
        - Remove diacritics | إزالة التشكيل
        - Normalize Hamza | تطبيع الهمزة
        - Normalize Alef Maksura | تطبيع الألف المقصورة
        - Remove Tatweel | إزالة التطويل
        
        Aggressive:
        - All standard normalizations | جميع تطبيعات standard
        - Normalize Taa Marbuta | تطبيع التاء المربوطة
        - Remove repeated characters | إزالة الأحرف المكررة
        
        Args:
            text: النص الأصلي / Original text
            
        Returns:
            النص المطبّع / Normalized text
        """
        original_text = text
        
        # Minimal: إزالة التشكيل فقط / Remove diacritics only
        text = self.ARABIC_NORMALIZATION_PATTERNS['diacritics'].sub('', text)
        if self.track_processing_stats:
            diacritics_removed = len(original_text) - len(text)
            self.processing_stats.removed_diacritics += diacritics_removed
        
        if self.normalization_level == 'minimal':
            return text
        
        # Standard: تطبيع الهمزات / Normalize Hamzas
        hamza_count = len(self.ARABIC_NORMALIZATION_PATTERNS['hamza_on_alef'].findall(text))
        text = self.ARABIC_NORMALIZATION_PATTERNS['hamza_on_alef'].sub('ا', text)
        text = self.ARABIC_NORMALIZATION_PATTERNS['hamza_on_waw'].sub('و', text)
        text = self.ARABIC_NORMALIZATION_PATTERNS['hamza_on_ya'].sub('ي', text)
        
        if self.track_processing_stats:
            self.processing_stats.normalized_hamzas += hamza_count
        
        # تطبيع الألف المقصورة / Normalize Alef Maksura
        text = self.ARABIC_NORMALIZATION_PATTERNS['alef_maksura'].sub('ي', text)
        
        # إزالة التطويل / Remove Tatweel
        text = self.ARABIC_NORMALIZATION_PATTERNS['tatweel'].sub('', text)
        
        if self.normalization_level == 'standard':
            return text
        
        # Aggressive: تطبيع التاء المربوطة / Normalize Taa Marbuta
        text = self.ARABIC_NORMALIZATION_PATTERNS['taa_marbuta'].sub('ه', text)
        
        return text
    
    @property
    def embedding_dim(self) -> int:
        """
        Get embedding dimension
        
        Returns:
            بُعد المتجهات / Embedding dimension
        """
        return self.model.get_sentence_embedding_dimension()
    
    def save_embeddings(self, texts: List[str], filepath: str, 
                       save_texts: bool = True,
                       save_metadata: bool = True) -> None:
        """
        Save embeddings to file
        
        Args:
            texts: Texts to embed
            filepath:  File path
            save_texts:  Also save texts
            save_metadata:  Save additional metadata
        """
        if not texts:
            raise ValueError("❌ Cannot save embeddings for empty texts list")

        # تأكد إن النصوص قائمة دائماً / Always treat as list
        texts_list = [texts] if isinstance(texts, str) else list(texts)

        # encode بترجع 1D لو نص واحد — نضمن 2D للحفظ الصحيح
        # encode returns 1D for single text — ensure 2D for consistent saving
        embeddings = self.encode(texts_list)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        save_dict = {'embeddings': embeddings}

        if save_texts:
            save_dict['texts'] = np.array(texts_list, dtype=object)

        if save_metadata:
            # np.savez بتقبل arrays بس — نلف الـ dict في np.array
            # np.savez only accepts arrays — wrap dict in np.array
            save_dict['metadata'] = np.array({
                'model_name': self.model_name,
                'model_key': self.model_key,
                'embedding_dim': self.embedding_dim,
                'normalization_level': self.normalization_level,
                'preprocessing_enabled': self.enable_preprocessing,
                'num_texts': len(texts_list),
                'timestamp': datetime.now().isoformat()
            })

        # np.savez بتضيف .npz لوحدها — نستخدم Path لتوحيد المسار
        # np.savez always appends .npz — use Path to normalize
        filepath = Path(filepath)
        if filepath.suffix == '.npz':
            filepath = filepath.with_suffix('')  # شيل .npz عشان numpy متضيفهاش تاني

        np.savez(str(filepath), **save_dict)  # هيحفظ كـ filepath.npz

        saved_path = filepath.with_suffix('.npz')
        file_size = saved_path.stat().st_size / 1024
        logger.info(
            f"💾 Saved {len(texts_list)} embeddings → {saved_path}\n"
            f"   Shape : {embeddings.shape}\n"
            f"   Size  : {file_size:.1f} KB"
        )
    
    def load_embeddings(self, filepath: str, 
                       load_texts: bool = False,
                       load_metadata: bool = False) -> Union[np.ndarray, Tuple]:
        """
        Load embeddings from file
        
        Args:
            filepath:  File path
            load_texts:  Also load texts
            load_metadata:  Load metadata
            
        Returns:
            Embeddings or (embeddings, texts) or (embeddings, texts, metadata)
        """
        filepath = str(filepath)
        if not filepath.endswith('.npz'):
            filepath += '.npz'

        if not Path(filepath).exists():
            raise FileNotFoundError(f"❌ File not found: {filepath}")

        try:
            data = np.load(filepath, allow_pickle=True)

            # التحقق من وجود الـ embeddings / Validate embeddings key exists
            if 'embeddings' not in data:
                raise KeyError(
                    f"❌ 'embeddings' key not found in file.\n"
                    f"   Available keys: {list(data.keys())}\n"
                    f"   The file may have been saved incorrectly."
                )

            embeddings = data['embeddings']

            # ضمان 2D shape / Ensure 2D shape
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)

            result = [embeddings]

            if load_texts:
                if 'texts' not in data:
                    logger.warning("⚠️ 'texts' key not found in file — was save_texts=True when saving?")
                else:
                    texts = data['texts']
                    result.append(texts)
                    logger.info(f"📂 Loaded {len(texts)} texts")

            if load_metadata:
                if 'metadata' not in data:
                    logger.warning("⚠️ 'metadata' key not found in file — was save_metadata=True when saving?")
                else:
                    metadata = data['metadata'].item()  # 0-d object array → dict
                    result.append(metadata)
                    logger.info(
                        f"📂 Loaded metadata:\n"
                        f"   Model     : {metadata.get('model_name', 'unknown')}\n"
                        f"   Num texts : {metadata.get('num_texts', 'unknown')}\n"
                        f"   Saved at  : {metadata.get('timestamp', 'unknown')}"
                    )

            logger.info(
                f"✅ Loaded embeddings from {filepath}\n"
                f"   Shape : {embeddings.shape}"
            )

            return tuple(result) if len(result) > 1 else result[0]

        except Exception as e:
            logger.error(f"❌ Error loading file: {e}")
            raise
    
    def compute_similarity(self, text1: Union[str, np.ndarray], 
                          text2: Union[str, np.ndarray],
                          metric: str = 'cosine') -> float:
        """
        Compute similarity between two texts
        
        Args:
            text1:  First text or its embedding
            text2: Second text or its embedding
            metric: Similarity metric (cosine/euclidean/dot)
        
        Returns:
            Similarity score
        """
        # الحصول على المتجهات / Get embeddings
        emb1 = self.encode(text1) if isinstance(text1, str) else text1
        emb2 = self.encode(text2) if isinstance(text2, str) else text2
        
        # التأكد من أن المتجهات أحادية البعد / Ensure 1D arrays
        emb1 = emb1.flatten() if emb1.ndim > 1 else emb1
        emb2 = emb2.flatten() if emb2.ndim > 1 else emb2
        
        if metric == 'cosine':
            # تشابه جيبي / Cosine similarity
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            if norm1 == 0 or norm2 == 0:
                similarity = 0.0
            else:
                similarity = np.dot(emb1, emb2) / (norm1 * norm2)
        elif metric == 'euclidean':
            # المسافة الإقليدية (معكوسة) / Euclidean distance (inverted)
            similarity = 1 / (1 + np.linalg.norm(emb1 - emb2))
        elif metric == 'dot':
            # الضرب النقطي / Dot product
            similarity = np.dot(emb1, emb2)
        else:
            raise ValueError(f"❌ Unknown metric: {metric}")
        
        return float(similarity)
    
    def find_most_similar(self, query: str, 
                         candidates: List[str],
                         top_k: int = 5,
                         metric: str = 'cosine') -> List[Tuple[int, str, float]]:
        """
        Find most similar texts
        
        Args:
            query:  Query text
            candidates: Candidate texts
            top_k: Number of results
            metric: Similarity metric
            
        Returns:
            List of (index, text, score)
        """
        # تضمين الاستعلام / Encode query
        query_emb = self.encode(query)
        
        # تضمين المرشحين / Encode candidates
        candidate_embs = self.encode(candidates)
        
        # حساب التشابه / Compute similarities
        similarities = []
        for i, cand_emb in enumerate(candidate_embs):
            sim = self.compute_similarity(query_emb, cand_emb, metric)
            similarities.append((i, candidates[i], sim))
        
        # ترتيب حسب التشابه / Sort by similarity
        similarities.sort(key=lambda x: x[2], reverse=True)
        
        return similarities[:top_k]
    
    def benchmark(self, sample_texts: Optional[List[str]] = None, 
                  num_iterations: int = 10,
                  warmup_iterations: int = 2) -> Dict:
        """
        Benchmark model performance
        
        Args:
            sample_texts: Sample texts
            num_iterations: Number of iterations
            warmup_iterations: Warmup iterations
        
        Returns:
            Performance statistics
        """
        # نصوص افتراضية / Default texts
        if sample_texts is None:
            sample_texts = [
                "هذا نص تجريبي للقياس",
                "الذكاء الاصطناعي يغير العالم",
                "البرمجة مهارة مهمة في العصر الحديث",
                "تطوير البرمجيات يتطلب مهارات متعددة",
                "التعلم الآلي جزء من الذكاء الاصطناعي",
                "البيانات الضخمة تساعد في اتخاذ القرارات",
                "الحوسبة السحابية توفر المرونة",
                "الأمن السيبراني مهم في العصر الرقمي"
            ]
        
        logger.info(
            f"🔬 Starting benchmark\n"
            f"   Texts: {len(sample_texts)}\n"
            f"   Iterations: {num_iterations}\n"
            f"   Warmup: {warmup_iterations}"
        )
        
        # إحماء / Warmup
        for _ in range(warmup_iterations):
            self.encode(sample_texts)
        
        # القياس / Benchmark
        times = []
        for i in range(num_iterations):
            start = time.time()
            embeddings = self.encode(sample_texts)
            elapsed = time.time() - start
            times.append(elapsed)
            logger.debug(f"Iteration {i+1}/{num_iterations}: {elapsed:.4f}s")
        
        # الإحصائيات / Statistics
        stats = {
            'num_texts': len(sample_texts),
            'num_iterations': num_iterations,
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'median_time': np.median(times),
            'texts_per_second': len(sample_texts) / np.mean(times),
            'embedding_dim': self.embedding_dim,
            'device': str(self.device),
            'model_name': self.model_name,
            'preprocessing_enabled': self.enable_preprocessing
        }
        
        logger.info(
            f"✅ Benchmark complete\n"
            f"   Average: {stats['avg_time']:.4f}s\n"
            f"   Texts/sec: {stats['texts_per_second']:.2f}\n"
            f"   Device: {stats['device']}"
        )
        
        return stats
    
    def get_model_info(self) -> Dict:
        """
        Get model information
        
        Returns:
            Model information
        """
        info = {
            'model_name': self.model_name,
            'model_key': self.model_key,
            'model_class': self.__class__.__name__,
            'embedding_dim': self.embedding_dim,
            'max_seq_length': self.max_seq_length,
            'device': str(self.device),
            'normalize_embeddings': self.normalize_embeddings,
            'preprocessing_enabled': self.enable_preprocessing,
            'normalization_level': self.normalization_level,
            'batch_size': self.batch_size
        }
        
        if self.model_info:
            info['model_description_ar'] = self.model_info.get('description_ar')
            info['model_description_en'] = self.model_info.get('description_en')
            info['best_for'] = self.model_info.get('best_for')
        
        if self.track_processing_stats:
            info['processing_stats'] = self.processing_stats.get_summary()
        
        return info
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics
        
        Returns:
            Processing statistics
        """
        if not self.track_processing_stats:
            return {'tracking_enabled': False}
        
        return self.processing_stats.get_summary()
    
    def reset_stats(self):
        """
        Reset statistics
        """
        if self.track_processing_stats:
            self.processing_stats = ArabicProcessingStats()
            logger.info("📊 Statistics reset")
    
    @staticmethod
    def list_recommended_models() -> Dict[str, Dict]:
        """
        List recommended models
        
        Returns:
            Dictionary of recommended models
        """
        return ArabicEmbedder.RECOMMENDED_MODELS
    
    def __enter__(self):
        """Support context manager"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """ Cleanup on exit"""
        if self.track_processing_stats:
            stats = self.processing_stats.get_summary()
            logger.info(
                f"\n📊 Session Summary:\n"
                f"   Total texts: {stats['total_texts']}\n"
                f"   Normalized: {stats['normalized_texts']}\n"
                f"   Norm rate: {stats['normalization_rate']:.1%}\n"
                f"   Elapsed: {stats['elapsed_time_seconds']:.1f}s"
            )
        self.cleanup()
        return False
    
    def __repr__(self) -> str:
        """String representation"""
        model_display = self.model_key or self.model_name.split('/')[-1]
        return (
            f"{self.__class__.__name__}("
            f"model={model_display}, "
            f"dim={self.embedding_dim}, "
            f"device={self.device}, "
            f"norm_level={self.normalization_level})"
        )