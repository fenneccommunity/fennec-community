
from .arabic_embedder import ArabicEmbedder
from .base_embedder import BaseEmbedder
from .config_embedder import EmbedderConfig
from .gemini_embedder import GeminiEmbedder
from .hugginface_embedder import HuggingFaceEmbedder 
from .ollama_embedder import OllamaEmbedder
from .mistral_embedder import MistralEmbedder
from .openai_embedder import OpenAIEmbedder



__all__ = ["ArabicEmbedder", "BaseEmbedder","EmbedderConfig", "GeminiEmbedder", "HuggingFaceEmbedder","EmbedderMode","OllamaEmbedder","MistralEmbedder","OpenAIEmbedder"]

__arabic_normalization__ = [
    'hamza_on_alef',
    'hamza_on_waw',
    'hamza_on_ya',
    'alef_maksura',
    'taa_marbuta',
    'diacritics',  
    'tatweel',
    'repeated_chars',  
]


__valid_levels__ = ['minimal : Remove diacritics only ',
                     'standard : Remove diacritics, normalize alef, ya, and taa marbuta but keep hamza and tatweel ', 
                     'aggressive : Remove diacritics, normalize alef, ya, taa marbuta, hamza, tatweel, and reduce repeated characters ']


__task_types_gemini__ = {
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

__gemini_embedding_models__ = {
        "text-embedding-004": {
            "dimensions": 768,
            "max_tokens": 2048,
            "max_batch_size": 100,
            "output_dimensionality_support": True,
            "min_output_dimensionality": 1,
            "arabic_support": "excellent",
            "languages": "100+",
            "release_date": "2024"
        },
        "embedding-001": {
            "dimensions": 768,
            "max_tokens": 2048,
            "max_batch_size": 100,
            "output_dimensionality_support": False,
            "arabic_support": "excellent",
            "languages": "100+",
            "release_date": "2023"
        },
        "models/text-embedding-004": {
            "dimensions": 768,
            "max_tokens": 2048,
            "max_batch_size": 100,
            "output_dimensionality_support": True,
            "min_output_dimensionality": 1,
            "arabic_support": "excellent",
            "languages": "100+"
        },
        "models/embedding-001": {
            "dimensions": 768,
            "max_tokens": 2048,
            "max_batch_size": 100,
            "output_dimensionality_support": False,
            "arabic_support": "excellent",
            "languages": "100+"
        }
    }

