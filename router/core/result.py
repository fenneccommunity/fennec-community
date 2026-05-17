"""

Typed result objects produced by the routing pipeline.
Rich enough for debugging, yet thin enough not to be in the hot path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import HandlerResponse
from ..config import ConfidenceLevel


# ---------------------------------------------------------------------------
# Candidate (pre-execution scoring record)
# ---------------------------------------------------------------------------

@dataclass
class RouteCandidate:
    """A scored route candidate surfaced by the RoutingPipeline."""
    route_name:      str
    group_name:      Optional[str]
    combined_score:  float
    semantic_score:  float
    keyword_score:   float
    llm_score:       float
    confidence:      ConfidenceLevel
    score_bias:      float = 0.0

    @property
    def effective_score(self) -> float:
        return min(1.0, self.combined_score + self.score_bias)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "route":          self.route_name,
            "group":          self.group_name,
            "score":          round(self.effective_score, 4),
            "semantic":       round(self.semantic_score, 4),
            "keyword":        round(self.keyword_score, 4),
            "llm":            round(self.llm_score, 4),
            "confidence":     self.confidence.value,
        }


# ---------------------------------------------------------------------------
# Routing Trace  (the full decision audit trail)
# ---------------------------------------------------------------------------

@dataclass
class RoutingTrace:
    """
    Full audit trail of a single routing decision.
    Emitted by the pipeline regardless of whether a match was found.
    """
    request_id:        str
    query:             str
    candidates:        List[RouteCandidate]  = field(default_factory=list)
    selected_group:    Optional[str]         = None
    selected_route:    Optional[str]         = None
    confidence:        Optional[ConfidenceLevel] = None
    used_llm_fallback: bool                  = False
    used_cache:        bool                  = False
    embedding_time_ms: float                 = 0.0
    scoring_time_ms:   float                 = 0.0
    execution_time_ms: float                 = 0.0
    total_time_ms:     float                 = 0.0
    notes:             List[str]             = field(default_factory=list)

    def add_note(self, note: str):
        self.notes.append(note)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id":        self.request_id,
            "query":             self.query[:120],
            "selected_group":    self.selected_group,
            "selected_route":    self.selected_route,
            "confidence":        self.confidence.value if self.confidence else None,
            "used_llm_fallback": self.used_llm_fallback,
            "used_cache":        self.used_cache,
            "timing_ms": {
                "embedding":  round(self.embedding_time_ms, 2),
                "scoring":    round(self.scoring_time_ms, 2),
                "execution":  round(self.execution_time_ms, 2),
                "total":      round(self.total_time_ms, 2),
            },
            "candidates": [c.to_dict() for c in self.candidates[:5]],
            "notes":      self.notes,
        }


# ---------------------------------------------------------------------------
# RoutingResult  (the object returned to the caller)
# ---------------------------------------------------------------------------

@dataclass
class RoutingResult:
    """
    The final result object returned by ``HierarchicalRouter.route()``.

    Contains both the handler's response and the full routing trace,
    so callers can inspect *why* a route was chosen without additional calls.
    """
    response:    Optional[HandlerResponse]
    trace:       RoutingTrace
    matched:     bool  = False

    # ---- Convenience accessors ------------------------------------------

    @property
    def content(self) -> Any:
        """Short-cut to ``response.content`` — the most common access."""
        return self.response.content if self.response else None

    @property
    def success(self) -> bool:
        return bool(self.matched and self.response and self.response.success)

    @property
    def route_name(self) -> Optional[str]:
        return self.trace.selected_route

    @property
    def group_name(self) -> Optional[str]:
        return self.trace.selected_group

    @property
    def confidence(self) -> Optional[ConfidenceLevel]:
        return self.trace.confidence

    @property
    def total_time_ms(self) -> float:
        return self.trace.total_time_ms

    # ---- Serialisation --------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "matched":   self.matched,
            "success":   self.success,
            "route":     self.route_name,
            "group":     self.group_name,
            "confidence": self.confidence.value if self.confidence else None,
            "content":   self.content,
            "error":     self.response.error if self.response else None,
            "time_ms":   round(self.total_time_ms, 2),
            "trace":     self.trace.to_dict(),
        }

    def __bool__(self) -> bool:
        return self.success

    def __repr__(self) -> str:
        if self.matched:
            return (
                f"RoutingResult(route={self.route_name!r}, "
                f"group={self.group_name!r}, "
                f"confidence={self.confidence}, "
                f"time={self.total_time_ms:.1f}ms, success={self.success})"
            )
        return (
            f"RoutingResult(no_match, time={self.total_time_ms:.1f}ms)"
        )


# ---------------------------------------------------------------------------
# Multi-route result (parallel / sequential execution)
# ---------------------------------------------------------------------------

@dataclass
class MultiRoutingResult:
    """Wraps results from multi-route (top-k) execution."""
    results:          List[RoutingResult]
    aggregated:       Optional[HandlerResponse]  = None
    trace:            Optional[RoutingTrace]     = None

    @property
    def success(self) -> bool:
        return bool(self.aggregated and self.aggregated.success)

    @property
    def content(self) -> Any:
        return self.aggregated.content if self.aggregated else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success":    self.success,
            "content":    self.content,
            "results":    [r.to_dict() for r in self.results],
            "trace":      self.trace.to_dict() if self.trace else None,
        }
