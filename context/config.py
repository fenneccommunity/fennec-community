"""
Advanced Configuration for Context Intelligence Engine
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


# Type alias
LengthCounter = Callable[[str], int]


@dataclass
class QueryAnalyzerConfig:
    """query analyzer settings"""
    # حجم السياق المخصص لكل مستوى تعقيد
    budget_low:    int = 800
    budget_medium: int = 2000
    budget_high:   int = 4000

    # قائمة الكيانات / المجالات المخصصة
    custom_entities: List[str] = field(default_factory=list)

    # عتبة تصنيف اللغة العربية (نسبة الأحرف العربية)
    arabic_threshold: float = 0.3


@dataclass
class RetrieverConfig:
    """retriever settings"""
    top_k:                int   = 20     # عدد النتائج المُسترجعة (قبل الفلترة)
    vector_weight:        float = 0.60   # وزن نتائج vector search
    keyword_weight:       float = 0.30   # وزن نتائج keyword search
    metadata_weight:      float = 0.10   # وزن نتائج metadata filtering
    use_vector:           bool  = True
    use_keyword:          bool  = True
    use_metadata:         bool  = False
    metadata_filters:     Dict[str, str] = field(default_factory=dict)
    min_vector_score:     float = 0.0    # حد أدنى للقبول من vector search


@dataclass
class FilterConfig:
    """filter settings"""
    dedup_method:         str   = "hash"    # "hash" | "prefix" | "semantic"
    prefix_length:        int   = 100
    semantic_sim_threshold: float = 0.92   # عتبة التشابه لاعتبار chunk مكرر دلالياً
    min_chunk_length:     int   = 20       # حذف الـ chunks القصيرة جداً
    max_chunk_length:     int   = 4000     # حذف الـ chunks الضخمة جداً
    noise_patterns:       List[str] = field(default_factory=lambda: [
        r"^\s*$",                        # فارغ
        r"^[\.\-\_\*]{5,}$",            # خطوط فاصلة
        r"^\s*(page|صفحة)\s*\d+\s*$",   # أرقام صفحات
    ])


@dataclass
class RankingConfig:
    """ranking settings"""
    # weights for the total score components
    weight_vector_score:   float = 0.40
    weight_keyword_score:  float = 0.25
    weight_source_quality: float = 0.20
    weight_recency:        float = 0.10
    weight_position:       float = 0.05   # هل الـ chunk في بداية المستند؟
    max_chunks_to_rank:    int   = 15     # أقصى عدد يُمرَّر للـ Composer


@dataclass
class ComposerConfig:
    """إعدادات Context Composer"""
    max_context_length:   int   = 2000    # الحد الأقصى للسياق الناتج
    separator:            str   = "\n---\n"
    template:             str   = "arabic"  # "arabic" | "english" | "minimal" | "structured"
    include_scores:       bool  = False
    include_metadata:     bool  = False
    include_sources:      bool  = True
    group_by_source:      bool  = False   # تجميع الـ chunks من نفس المصدر
    summarize_long_chunks:bool  = False   # تلخيص الـ chunks الطويلة (يتطلب LLM)
    min_compression_chars:int   = 50


@dataclass
class GuardConfig:
    """إعدادات Context Guard"""
    enabled:                  bool  = True
    min_context_length:       int   = 10     # حد أدنى لطول السياق
    max_context_length:       int   = 8000   # حد أقصى (تحذير)
    check_empty:              bool  = True
    check_relevance:          bool  = False  # يتطلب embedding (اختياري)
    relevance_threshold:      float = 0.10
    check_source_coverage:    bool  = True   # هل يوجد مصدر واحد على الأقل؟


@dataclass
class CacheConfig:
    """إعدادات الـ Cache"""
    enabled:      bool = True
    max_size:     int  = 256      # عدد الاستعلامات المُخزَّنة
    ttl_seconds:  int  = 300      # مدة الصلاحية


@dataclass
class ContextEngineConfig:
    """
   main configuration class for the entire context engine, aggregating all sub-component configs and global settings.
    """
    # ── Sub-component configs ──────────────────────────────────────────────
    query_analyzer: QueryAnalyzerConfig = field(default_factory=QueryAnalyzerConfig)
    retriever:      RetrieverConfig     = field(default_factory=RetrieverConfig)
    filter_cfg:     FilterConfig        = field(default_factory=FilterConfig)
    ranking:        RankingConfig       = field(default_factory=RankingConfig)
    composer:       ComposerConfig      = field(default_factory=ComposerConfig)
    guard:          GuardConfig         = field(default_factory=GuardConfig)
    cache:          CacheConfig         = field(default_factory=CacheConfig)

    # ── Global settings ────────────────────────────────────────────────────
    language:       str  = "auto"        # "auto" | "ar" | "en"
    enable_adaptive: bool = True         # التعلم التكيّفي
    multi_turn:      bool = True         # دعم المحادثات المتعددة

    # ── Legacy compatibility ───────────────────────────────────────────────
    @property
    def max_context_length(self) -> int:
        return self.composer.max_context_length

    @max_context_length.setter
    def max_context_length(self, v: int):
        self.composer.max_context_length = v

    @property
    def separator(self) -> str:
        return self.composer.separator

    @property
    def template(self) -> str:
        return self.composer.template


# ── Backward-compatible alias ──────────────────────────────────────────────
ContextConfig = ContextEngineConfig
