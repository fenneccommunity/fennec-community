"""

All configuration dataclasses for the hierarchical router system.
Centralised here so operators can tune the entire system from one place.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class SimilarityMetric(str, Enum):
    COSINE     = "cosine"
    DOT_PRODUCT = "dot_product"
    EUCLIDEAN  = "euclidean"


class ExecutionMode(str, Enum):
    SINGLE     = "single"      # Return the single best route
    SEQUENTIAL = "sequential"  # Execute top-k in order, stop on first success
    PARALLEL   = "parallel"    # Execute all candidates concurrently


class AggregationStrategy(str, Enum):
    FIRST_WINS = "first_wins"  # Return first successful response
    VOTING     = "voting"      # Majority vote on structured responses
    MERGE      = "merge"       # Merge all responses into one


class ConfidenceLevel(str, Enum):
    HIGH   = "high"    # ≥ high_threshold  → route directly
    MEDIUM = "medium"  # ≥ low_threshold   → route with warning
    LOW    = "low"     # ≥ fallback_threshold → LLM fallback
    NONE   = "none"    # below all thresholds → global fallback


class LogLevel(str, Enum):
    DEBUG   = "DEBUG"
    INFO    = "INFO"
    WARNING = "WARNING"
    ERROR   = "ERROR"


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingConfig:
    """Controls the embedding model used for semantic scoring."""
    model_name: str         = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    batch_size: int         = 32
    cache_embeddings: bool  = True          # Cache computed embeddings to disk/memory
    normalize_embeddings: bool = True       # L2-normalise before similarity


@dataclass
class ScoringConfig:
    """Weights and thresholds for the hybrid scorer."""
    semantic_weight: float  = 0.70          # Weight for embedding similarity
    keyword_weight: float   = 0.20          # Weight for keyword signals
    llm_weight: float       = 0.10          # Weight for optional LLM scorer

    similarity_metric: SimilarityMetric = SimilarityMetric.COSINE

    # Confidence thresholds (applied to the *combined* score)
    high_confidence_threshold: float    = 0.85
    medium_confidence_threshold: float  = 0.60
    fallback_confidence_threshold: float = 0.40

    top_k: int = 5                          # Number of candidate routes to surface


@dataclass
class ExecutionConfig:
    """Controls how matched routes are executed."""
    mode: ExecutionMode                  = ExecutionMode.SINGLE
    aggregation: AggregationStrategy     = AggregationStrategy.FIRST_WINS
    parallel_timeout: float              = 10.0   # seconds
    max_retries: int                     = 2
    retry_delay: float                   = 0.5    # seconds between retries
    retry_on_exceptions: List[str]       = field(
        default_factory=lambda: ["RuntimeError", "TimeoutError"]
    )


@dataclass
class CacheConfig:
    """Controls the layered cache."""
    enabled: bool           = True
    ttl_seconds: int        = 300           # Time-to-live for route decisions
    max_size: int           = 1_000         # Max entries in the LRU cache
    embedding_cache: bool   = True          # Also cache query embeddings
    embedding_ttl: int      = 3_600         # Embeddings are more stable → longer TTL


@dataclass
class FeedbackConfig:
    """Controls the adaptive feedback loop."""
    enabled: bool               = True
    score_decay: float          = 0.05      # How much to shift score per feedback tick
    min_samples_to_adapt: int   = 10        # Ignore routes with fewer calls
    persist_path: Optional[str] = None      # JSON path to save learned weights


@dataclass
class ObservabilityConfig:
    """Controls logging, tracing, and metrics."""
    log_level: LogLevel         = LogLevel.INFO
    structured_logging: bool    = True      # Emit JSON log lines
    enable_tracing: bool        = True      # Record full routing decision trace
    metrics_window: int         = 1_000     # Rolling window size for metrics
    slow_route_ms: float        = 500.0     # Latency threshold for warnings


# ---------------------------------------------------------------------------
# Master config
# ---------------------------------------------------------------------------

@dataclass
class RouterConfig:
    """Single entry-point for all router configuration."""
    embedding:      EmbeddingConfig     = field(default_factory=EmbeddingConfig)
    scoring:        ScoringConfig       = field(default_factory=ScoringConfig)
    execution:      ExecutionConfig     = field(default_factory=ExecutionConfig)
    cache:          CacheConfig         = field(default_factory=CacheConfig)
    feedback:       FeedbackConfig      = field(default_factory=FeedbackConfig)
    observability:  ObservabilityConfig = field(default_factory=ObservabilityConfig)

    # LLM fallback (optional)
    llm_model: Optional[str]           = None   # e.g. "gpt-4o-mini" or local endpoint
    llm_api_key: Optional[str]         = None

    # Global fallback behaviour when no route matches
    raise_on_no_match: bool            = False  # If True, raise; else use fallback handler

    @classmethod
    def fast(cls) -> "RouterConfig":
        """Preset: optimised for low latency (single route, cache on)."""
        cfg = cls()
        cfg.scoring.top_k = 1
        cfg.execution.mode = ExecutionMode.SINGLE
        cfg.cache.enabled = True
        return cfg

    @classmethod
    def accurate(cls) -> "RouterConfig":
        """Preset: optimised for accuracy (parallel top-3, LLM fallback)."""
        cfg = cls()
        cfg.scoring.top_k = 3
        cfg.execution.mode = ExecutionMode.PARALLEL
        cfg.scoring.llm_weight = 0.20
        cfg.scoring.semantic_weight = 0.60
        return cfg
