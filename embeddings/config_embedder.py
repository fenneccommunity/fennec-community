"""
Embedder Configuration 
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class EmbedderConfig:
    """Main Embedding Configuration"""
    
    # إعدادات النموذج
    model_name: str = "multilingual"
    device: Optional[str] = None
    cache_dir: Optional[str] = None
    
    # إعدادات التضمين
    batch_size: int = 32
    normalize_embeddings: bool = True
    max_seq_length: int = 512
    
    # إعدادات المعالجة
    enable_preprocessing: bool = True
    enable_arabic_normalization: bool = True
    
    # إعدادات الذاكرة المؤقتة (للـ CachedEmbedder)
    enable_cache: bool = False
    cache_size: int = 10000
    
    # إعدادات الأداء
    show_progress_bar: bool = False
    convert_to_numpy: bool = True
    skip_preprocessing: bool = False
    return_valid_indices: bool = False
    
    # إعدادات إضافية
    extra_config: Dict[str, Any] = field(default_factory=dict)
    
    # إعدادات الإحصائيات
    track_processing_stats: bool = True

    auto_download: bool = True
    trust_remote_code: bool = False

    # ollama
    base_url: str = "http://127.0.0.1:11434"
    host: str = "localhost"
    port: int = 11434
    embedding_model : str = "nomic-embed-text"

    