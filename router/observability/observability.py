"""

Structured logging, per-request tracing, and rolling metrics collection.

Components
----------
RouterLogger   : Wraps Python logging with structured JSON output and
                 correlation IDs.
MetricsCollector : Rolling-window metrics for latency, confidence,
                   hit rates, and per-route stats.
"""
from __future__ import annotations

import collections
import json
import logging
import statistics
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List

from ..config import  ObservabilityConfig
from ..core.result import RoutingResult


# ---------------------------------------------------------------------------
# Structured Logger
# ---------------------------------------------------------------------------

class RouterLogger:
    """
    Thin wrapper over Python's ``logging`` module that emits structured
    JSON log lines when ``structured_logging=True``.

    Every log call accepts ``extra`` kwargs which are merged into the JSON blob,
    enabling rich context (request_id, route, latency…) without string formatting.
    """

    def __init__(self, cfg: ObservabilityConfig, name: str = "router_v2"):
        self._cfg  = cfg
        self._log  = logging.getLogger(name)
        level = getattr(logging, cfg.log_level.value, logging.INFO)
        self._log.setLevel(level)

        if not self._log.handlers:
            handler = logging.StreamHandler()
            if cfg.structured_logging:
                handler.setFormatter(_JsonFormatter())
            else:
                handler.setFormatter(logging.Formatter(
                    "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
                ))
            self._log.addHandler(handler)

    def info(self, msg: str, **extra):
        self._emit(logging.INFO, msg, extra)

    def debug(self, msg: str, **extra):
        self._emit(logging.DEBUG, msg, extra)

    def warning(self, msg: str, **extra):
        self._emit(logging.WARNING, msg, extra)

    def error(self, msg: str, **extra):
        self._emit(logging.ERROR, msg, extra)

    def _emit(self, level: int, msg: str, extra: Dict[str, Any]):
        if extra:
            self._log.log(level, msg, extra={"_ctx": extra})
        else:
            self._log.log(level, msg)

    # ---- Convenience routing-specific helpers ---------------------------

    def log_routing_decision(self, result: RoutingResult):
        trace = result.trace
        self.info(
            "routing_decision",
            request_id    = trace.request_id,
            route         = trace.selected_route,
            group         = trace.selected_group,
            confidence    = trace.confidence.value if trace.confidence else None,
            total_ms      = round(trace.total_time_ms, 2),
            embedding_ms  = round(trace.embedding_time_ms, 2),
            scoring_ms    = round(trace.scoring_time_ms, 2),
            execution_ms  = round(trace.execution_time_ms, 2),
            matched       = result.matched,
            success       = result.success,
            from_cache    = trace.used_cache,
        )
        if trace.total_time_ms > self._cfg.slow_route_ms:
            self.warning(
                "slow_route_detected",
                request_id = trace.request_id,
                route      = trace.selected_route,
                total_ms   = round(trace.total_time_ms, 2),
                threshold  = self._cfg.slow_route_ms,
            )

    def log_no_match(self, query: str, request_id: str, top_score: float):
        self.warning(
            "no_route_matched",
            request_id = request_id,
            query_len  = len(query),
            top_score  = round(top_score, 4),
        )

    def log_cache_hit(self, request_id: str, route_name: str):
        self.debug("cache_hit", request_id=request_id, route=route_name)

    def log_feedback(self, route_name: str, success: bool, score: float):
        self.debug(
            "feedback_recorded",
            route   = route_name,
            success = success,
            score   = round(score, 4),
        )


class _JsonFormatter(logging.Formatter):
    """Emit each log line as a single JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts":      self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level":   record.levelname,
            "logger":  record.name,
            "msg":     record.getMessage(),
        }
        ctx = getattr(record, "_ctx", None)
        if ctx:
            payload.update(ctx)
        return json.dumps(payload, default=str)


# ---------------------------------------------------------------------------
# Metrics Collector
# ---------------------------------------------------------------------------

@dataclass
class RouteStats:
    """Aggregated statistics for a single route."""
    calls:         int   = 0
    successes:     int   = 0
    failures:      int   = 0
    total_ms:      float = 0.0
    latencies_ms:  List[float] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.successes / self.calls if self.calls else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_ms / self.calls if self.calls else 0.0

    @property
    def p95_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_lat = sorted(self.latencies_ms)
        idx = max(0, int(len(sorted_lat) * 0.95) - 1)
        return sorted_lat[idx]


class MetricsCollector:
    """
    Rolling-window metrics for the router system.

    Thread-safe; uses deques with a fixed ``maxlen`` so memory is bounded.
    """

    def __init__(self, cfg: ObservabilityConfig):
        self._window = cfg.metrics_window
        self._lock   = threading.Lock()

        # Rolling windows
        self._latencies_ms:         Deque[float]          = collections.deque(maxlen=self._window)
        self._confidences:          Deque[str]            = collections.deque(maxlen=self._window)
        self._matches:              Deque[bool]           = collections.deque(maxlen=self._window)
        self._route_stats:          Dict[str, RouteStats] = {}

        # Cumulative totals (never reset)
        self._total_requests:  int   = 0
        self._cache_hits:      int   = 0
        self._llm_invocations: int   = 0
        self._start_time:      float = time.time()

    # ------------------------------------------------------------------ #
    # Recording
    # ------------------------------------------------------------------ #

    def record(self, result: RoutingResult):
        trace    = result.trace
        total_ms = trace.total_time_ms

        with self._lock:
            self._total_requests += 1
            self._latencies_ms.append(total_ms)
            self._matches.append(result.matched)

            if trace.confidence:
                self._confidences.append(trace.confidence.value)

            if trace.used_cache:
                self._cache_hits += 1

            if trace.used_llm_fallback:
                self._llm_invocations += 1

            if result.matched and trace.selected_route:
                route_name = trace.selected_route
                if route_name not in self._route_stats:
                    self._route_stats[route_name] = RouteStats()

                rs = self._route_stats[route_name]
                rs.calls += 1
                rs.total_ms += total_ms
                rs.latencies_ms.append(total_ms)
                # Trim per-route latency list to window size
                if len(rs.latencies_ms) > self._window:
                    rs.latencies_ms = rs.latencies_ms[-self._window:]
                if result.success:
                    rs.successes += 1
                else:
                    rs.failures += 1

    # ------------------------------------------------------------------ #
    # Snapshot
    # ------------------------------------------------------------------ #

    def snapshot(self) -> Dict[str, Any]:
        """Return a point-in-time metrics snapshot."""
        with self._lock:
            lats  = list(self._latencies_ms)
            match = list(self._matches)
            confs = list(self._confidences)

            avg_lat = statistics.mean(lats)   if lats   else 0.0
            p95_lat = sorted(lats)[max(0, int(len(lats) * 0.95) - 1)] if lats else 0.0
            match_rate = sum(match) / len(match) * 100 if match else 0.0
            cache_rate = (
                self._cache_hits / self._total_requests * 100
                if self._total_requests else 0.0
            )

            conf_dist = {}
            for c in confs:
                conf_dist[c] = conf_dist.get(c, 0) + 1

            route_summary = {
                name: {
                    "calls":       rs.calls,
                    "success_rate": round(rs.success_rate * 100, 2),
                    "avg_ms":      round(rs.avg_latency_ms, 2),
                    "p95_ms":      round(rs.p95_latency_ms, 2),
                }
                for name, rs in self._route_stats.items()
            }

            uptime_s = time.time() - self._start_time

            return {
                "uptime_seconds":    round(uptime_s, 1),
                "total_requests":    self._total_requests,
                "window_size":       self._window,
                "match_rate_pct":    round(match_rate, 2),
                "cache_hit_rate_pct": round(cache_rate, 2),
                "llm_invocations":   self._llm_invocations,
                "latency_ms": {
                    "avg":  round(avg_lat, 2),
                    "p95":  round(p95_lat, 2),
                    "min":  round(min(lats), 2)  if lats else 0.0,
                    "max":  round(max(lats), 2)  if lats else 0.0,
                },
                "confidence_distribution": conf_dist,
                "routes": route_summary,
            }

    def reset(self):
        with self._lock:
            self._latencies_ms.clear()
            self._confidences.clear()
            self._matches.clear()
            self._route_stats.clear()
            self._cache_hits      = 0
            self._llm_invocations = 0
            self._total_requests  = 0
            self._start_time      = time.time()
