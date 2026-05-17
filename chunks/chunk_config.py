"""
Configuration for the AI-Powered Chunking System
"""
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ChunkConfig:
    # ────────────────────────────────────────────
    # Basic sizing
    # ────────────────────────────────────────────
    chunk_size: int = 512
    overlap: int = 128
    min_chunk_size: int = 50
    max_chunk_size: int = 2048
    strict_size_limit: bool = True
    size_tolerance: float = 0.05          # 5% tolerance above chunk_size

    # ────────────────────────────────────────────
    # Semantic chunking
    # ────────────────────────────────────────────
    use_semantic_chunking: bool = False
    semantic_similarity_threshold: float = 0.75   # below → new chunk
    semantic_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    # alias kept for backward compat with ArabicTextChunker / MultilanguageTextChunker
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_batch_size: int = 64
    embedding_cache_size: int = 10_000    # LRU cache entries

    # ────────────────────────────────────────────
    # Adaptive chunking
    # ────────────────────────────────────────────
    use_adaptive_chunking: bool = False
    adaptive_technical_threshold: float = 0.6    # info-density → "technical"
    adaptive_min_size: int = 100
    adaptive_max_size: int = 1500
    adaptive_base_size: int = 512

    # ────────────────────────────────────────────
    # Smart overlap
    # ────────────────────────────────────────────
    use_smart_overlap: bool = True
    smart_overlap_threshold: float = 0.7

    # ────────────────────────────────────────────
    # Structure-aware chunking
    # ────────────────────────────────────────────
    respect_document_structure: bool = True
    split_on_headers: bool = True
    split_on_paragraphs: bool = True
    preserve_code_blocks: bool = True
    preserve_tables: bool = True
    preserve_lists: bool = True

    # ────────────────────────────────────────────
    # Context-aware / window merging
    # ────────────────────────────────────────────
    use_context_window: bool = False
    context_window_size: int = 2          # sentences on each side
    min_context_sentences: int = 3

    # ────────────────────────────────────────────
    # Metadata & scoring
    # ────────────────────────────────────────────
    extract_keywords: bool = True
    keyword_max_count: int = 10
    compute_chunk_scores: bool = True
    header_score_boost: float = 1.5       # multiply score for header chunks

    # ────────────────────────────────────────────
    # Deduplication
    # ────────────────────────────────────────────
    deduplication_enabled: bool = True
    dedup_similarity_threshold: float = 0.95   # cosine sim → duplicate
    dedup_use_hash: bool = True           # fast path: exact hash match

    # ────────────────────────────────────────────
    # Query-aware strategy
    # ────────────────────────────────────────────
    query_aware: bool = False
    default_query_pattern: str = "general"   # "faq" | "documentation" | "general"

    # ────────────────────────────────────────────
    # Performance
    # ────────────────────────────────────────────
    async_processing: bool = False
    n_workers: int = 4

    # ────────────────────────────────────────────
    # Formatting & text cleanup
    # ────────────────────────────────────────────
    preserve_formatting: bool = True
    fix_spacing: bool = True

    # ────────────────────────────────────────────
    # Tiktoken
    # ────────────────────────────────────────────
    model_token: str = "cl100k_base"

    # ────────────────────────────────────────────
    # Language-specific BERT models
    # ────────────────────────────────────────────
    arabic:      str = "CAMeL-Lab/bert-base-arabic-camelbert-mix"
    multilingual: str = "xlm-roberta-base"
    english:     str = "bert-base-uncased"
    chinese:     str = "bert-base-chinese"
    french:      str = "camembert-base"
    german:      str = "bert-base-german-cased"
    spanish:     str = "dccuchile/bert-base-spanish-wwm-uncased"
    russian:     str = "DeepPavlov/rubert-base-cased"
    japanese:    str = "cl-tohoku/bert-base-japanese"
    korean:      str = "klue/bert-base"
    portuguese:  str = "neuralmind/bert-base-portuguese-cased"
    italian:     str = "dbmdz/bert-base-italian-cased"
    dutch:       str = "GroNLP/bert-base-dutch-cased"
    polish:      str = "dkleczek/bert-base-polish-cased"
    turkish:     str = "dbmdz/bert-base-turkish-cased"
    vietnamese:  str = "vinai/phobert-base"
    hindi:       str = "ai4bharat/indic-bert"

    # ────────────────────────────────────────────
    # Transformer params
    # ────────────────────────────────────────────
    max_len: int = 512
    padding: bool = True
    truncation: bool = True
    return_tensor: str = "pt"

    # ────────────────────────────────────────────
    # Merge params
    # ────────────────────────────────────────────
    min_sentences: int = 30
