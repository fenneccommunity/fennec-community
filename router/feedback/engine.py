"""

The FeedbackEngine closes the routing loop by:

  1. Recording each routing decision's outcome (success / failure)
  2. Computing a per-route **score bias** that nudges the scorer toward or
     away from a route based on observed accuracy
  3. Optionally persisting the learned biases so they survive restarts

Score Adjustment Model
----------------------
We use an Exponential Moving Average (EMA) of success rate, converted
into a bias term in [-0.3, +0.3]:

    ema_n = alpha * outcome + (1 - alpha) * ema_{n-1}
    bias  = clip((ema - 0.5) * 0.6,  -0.3, +0.3)

  - ``alpha`` controls how quickly the system adapts (default 0.1)
  - At start, EMA = 0.5 (neutral) → bias = 0
  - After many successes: EMA → 1.0 → bias → +0.3 (route preferred)
  - After many failures:  EMA → 0.0 → bias → -0.3 (route penalised)

Only routes with at least ``min_samples`` observations are adjusted,
avoiding noise from rarely-used routes.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..core.route import Route
from ..config import FeedbackConfig


# ---------------------------------------------------------------------------
# Feedback Record
# ---------------------------------------------------------------------------

@dataclass
class FeedbackRecord:
    route_name:    str
    group_name:    Optional[str]
    query_hash:    str
    success:       bool
    score:         float
    timestamp:     float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Per-Route Feedback State
# ---------------------------------------------------------------------------

@dataclass
class RouteFeedbackState:
    route_name: str
    ema:        float = 0.5   # Start neutral
    n_samples:  int   = 0
    n_success:  int   = 0
    n_failure:  int   = 0

    @property
    def success_rate(self) -> float:
        return self.n_success / self.n_samples if self.n_samples > 0 else 0.0

    @property
    def computed_bias(self) -> float:
        return max(-0.3, min(0.3, (self.ema - 0.5) * 0.6))

    def update(self, success: bool, alpha: float):
        outcome = 1.0 if success else 0.0
        self.ema = alpha * outcome + (1 - alpha) * self.ema
        self.n_samples += 1
        if success:
            self.n_success += 1
        else:
            self.n_failure += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "route_name":   self.route_name,
            "ema":          round(self.ema, 6),
            "n_samples":    self.n_samples,
            "n_success":    self.n_success,
            "n_failure":    self.n_failure,
            "success_rate": round(self.success_rate, 4),
            "bias":         round(self.computed_bias, 4),
        }


# ---------------------------------------------------------------------------
# Feedback Engine
# ---------------------------------------------------------------------------

class FeedbackEngine:
    """
    Adaptive feedback loop that adjusts route scoring biases based on
    observed routing outcomes.

    Thread-safe for concurrent routing environments.

    Usage
    -----
    ::

        engine = FeedbackEngine(cfg)

        # After each routing decision:
        engine.record(route_name="rag.docs_qa", group="rag", success=True, score=0.87)

        # Apply learned biases to routes:
        engine.apply_biases(route_map)
    """

    def __init__(self, cfg: FeedbackConfig, alpha: float = 0.1):
        self._cfg    = cfg
        self._alpha  = alpha
        self._states: Dict[str, RouteFeedbackState] = {}
        self._history: List[FeedbackRecord]         = []
        self._lock   = threading.Lock()

        # Load persisted state if available
        if cfg.persist_path and os.path.exists(cfg.persist_path):
            self._load(cfg.persist_path)

    # ------------------------------------------------------------------ #
    # Recording
    # ------------------------------------------------------------------ #

    def record(
        self,
        route_name: str,
        group_name: Optional[str],
        success:    bool,
        score:      float,
        query:      str = "",
    ):
        """
        Record the outcome of a routing decision.

        Parameters
        ----------
        route_name : The route that was executed.
        group_name : The group the route belongs to (optional).
        success    : True if the handler returned a successful response.
        score      : The combined score at routing time.
        query      : Raw query (stored as hash only for privacy).
        """
        if not self._cfg.enabled:
            return

        import hashlib
        q_hash = hashlib.md5(query.encode()).hexdigest()[:8]

        with self._lock:
            if route_name not in self._states:
                self._states[route_name] = RouteFeedbackState(route_name=route_name)

            state = self._states[route_name]
            state.update(success, self._alpha)

            self._history.append(FeedbackRecord(
                route_name = route_name,
                group_name = group_name,
                query_hash = q_hash,
                success    = success,
                score      = score,
            ))

            # Trim history to last 10 000 records
            if len(self._history) > 10_000:
                self._history = self._history[-10_000:]

    # ------------------------------------------------------------------ #
    # Applying biases
    # ------------------------------------------------------------------ #

    def apply_biases(self, route_map: Dict[str, Route]):
        """
        Push learned bias values to all routes in the map.
        Should be called periodically (e.g. every N requests or on each request).
        """
        if not self._cfg.enabled:
            return

        min_samples = self._cfg.min_samples_to_adapt

        with self._lock:
            for route_name, state in self._states.items():
                if state.n_samples < min_samples:
                    continue   # Not enough data — don't adjust yet
                route = route_map.get(route_name)
                if route:
                    route.score_bias = state.computed_bias

    # ------------------------------------------------------------------ #
    # Bulk feedback (post-hoc labelling)
    # ------------------------------------------------------------------ #

    def bulk_record(self, records: List[Dict[str, Any]]):
        """
        Accept a batch of feedback events (e.g. from a labelling pipeline).

        Each record: {"route_name": str, "success": bool, "score": float}
        """
        for r in records:
            self.record(
                route_name = r["route_name"],
                group_name = r.get("group_name"),
                success    = r["success"],
                score      = r.get("score", 0.0),
                query      = r.get("query", ""),
            )

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: Optional[str] = None):
        """Persist learned states to JSON."""
        target = path or self._cfg.persist_path
        if not target:
            return
        with self._lock:
            data = {name: s.to_dict() for name, s in self._states.items()}
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _load(self, path: str):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            for name, d in data.items():
                s = RouteFeedbackState(route_name=name)
                s.ema       = d.get("ema", 0.5)
                s.n_samples = d.get("n_samples", 0)
                s.n_success = d.get("n_success", 0)
                s.n_failure = d.get("n_failure", 0)
                self._states[name] = s
        except (json.JSONDecodeError, OSError):
            pass   # Silently ignore corrupt / missing file

    # ------------------------------------------------------------------ #
    # Inspection
    # ------------------------------------------------------------------ #

    def get_state(self, route_name: str) -> Optional[RouteFeedbackState]:
        return self._states.get(route_name)

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "enabled":         self._cfg.enabled,
                "alpha":           self._alpha,
                "min_samples":     self._cfg.min_samples_to_adapt,
                "tracked_routes":  len(self._states),
                "total_records":   len(self._history),
                "states":          {n: s.to_dict() for n, s in self._states.items()},
            }

    def reset(self, route_name: Optional[str] = None):
        """Reset feedback state for one or all routes."""
        with self._lock:
            if route_name:
                self._states.pop(route_name, None)
            else:
                self._states.clear()
                self._history.clear()
