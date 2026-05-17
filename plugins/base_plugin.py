"""
BasePlugin — Unified Abstract Interface
=======================================
Every plugin in the ecosystem inherits from this class.
No arbitrary duck-typing; every capability is explicitly contracted here.
"""

from __future__ import annotations
import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime , UTC
from typing import Any, Dict, List, Optional

from .metadata import ExecutionContext, PluginMetadata, PluginStatus

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────

class PluginError(Exception):
    """Base exception for all plugin-related errors."""


class PluginTimeoutError(PluginError):
    """Raised when a plugin exceeds its configured timeout."""


class PluginValidationError(PluginError):
    """Raised when plugin input fails schema validation."""


class PluginPermissionError(PluginError):
    """Raised when a plugin attempts a forbidden operation."""


# ─────────────────────────────────────────────
# Plugin config  (kept slim — manager injects context)
# ─────────────────────────────────────────────

class PluginConfig:
    """Runtime configuration knobs for a plugin instance."""

    def __init__(
        self,
        enabled:        bool             = True,
        timeout:        float            = 30.0,
        max_retries:    int              = 3,
        cache_results:  bool             = False,
        cache_ttl_sec:  int              = 300,
        custom:         Dict[str, Any]   = None,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be > 0")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.enabled       = enabled
        self.timeout       = timeout
        self.max_retries   = max_retries
        self.cache_results = cache_results
        self.cache_ttl_sec = cache_ttl_sec
        self.custom        = custom or {}


# ─────────────────────────────────────────────
# Base Plugin
# ─────────────────────────────────────────────

class BasePlugin(ABC):
    """
    Abstract base for every plugin in the RAG ecosystem.

    Contract
    --------
    Subclasses MUST implement:
      - initialize()  → bool
      - execute(input_data, context) → Any
      - cleanup()     → None

    Subclasses SHOULD override:
      - validate(input_data) — default passes all inputs through
      - health_check()        — default returns True

    Everything else (stats, retries, timeouts, locking) is handled here.
    """

    # ── Class-level metadata; override in each plugin ──────────────────
    METADATA: PluginMetadata  # must be set by subclass at class level

    def __init__(self, config: Optional[PluginConfig] = None) -> None:
        if not hasattr(self.__class__, "METADATA"):
            raise TypeError(
                f"{self.__class__.__name__} must define a class-level METADATA attribute."
            )

        self.metadata   = self.__class__.METADATA
        self.name       = self.metadata.name
        self.config     = config or PluginConfig()
        self.status     = PluginStatus.DISABLED
        self.error_msg: Optional[str] = None

        # ── Execution statistics ──────────────────────────────────────
        self._exec_total:   int   = 0
        self._exec_success: int   = 0
        self._exec_failure: int   = 0
        self._time_total:   float = 0.0
        self._time_min:     float = float("inf")
        self._time_max:     float = 0.0
        self._time_last:    float = 0.0

        # ── Timeline ──────────────────────────────────────────────────
        self.initialized_at:  Optional[datetime] = None
        self.last_executed_at: Optional[datetime] = None
        self.last_error_at:    Optional[datetime] = None

        # ── Error journal ─────────────────────────────────────────────
        self._error_journal: List[Dict[str, Any]] = []
        self._MAX_JOURNAL    = 50

        # ── Result cache (optional) ───────────────────────────────────
        self._result_cache:  Dict[str, Any]  = {}
        self._cache_ts:      Dict[str, float] = {}

        # ── Concurrency guard ─────────────────────────────────────────
        self._lock = asyncio.Lock()

    # ─────────────────────────────────────────
    # Abstract interface
    # ─────────────────────────────────────────

    @abstractmethod
    async def initialize(self) -> bool:
        """
        Called once by the PluginManager after registration.
        Acquire resources (DB connections, model loads, etc.).
        Return True on success, False on failure.
        """

    @abstractmethod
    async def execute(self, input_data: Dict[str, Any], context: ExecutionContext) -> Any:
        """
        Core plugin logic.

        Parameters
        ----------
        input_data : dict
            Validated input matching the plugin's ``input_schema``.
        context : ExecutionContext
            Runtime context: query, session, memory, cache handles, etc.

        Returns
        -------
        Any
            Output matching the plugin's ``output_schema``.
        """

    @abstractmethod
    async def cleanup(self) -> None:
        """Release all resources acquired in initialize()."""

    # ─────────────────────────────────────────
    # Optional overrides
    # ─────────────────────────────────────────

    async def validate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and coerce input_data against the plugin's input_schema.
        Override to add custom validation logic.
        Raise PluginValidationError on failure.
        """
        required_fields = [
            p.name for p in self.metadata.input_schema if p.required
        ]
        missing = [f for f in required_fields if f not in input_data]
        if missing:
            raise PluginValidationError(
                f"[{self.name}] Missing required fields: {missing}"
            )
        return input_data

    async def health_check(self) -> bool:
        """Return True if the plugin is healthy and ready to serve."""
        return self.status == PluginStatus.ENABLED

    # ─────────────────────────────────────────
    # Public execution surface (used by manager)
    # ─────────────────────────────────────────

    async def safe_execute(
        self,
        input_data: Dict[str, Any],
        context: ExecutionContext,
    ) -> Any:
        """
        Validate → (optional cache lookup) → execute with timeout.
        Updates stats on every call.
        """
        if not self.config.enabled:
            raise PluginError(f"[{self.name}] Plugin is disabled.")
        if self.status == PluginStatus.ERROR:
            raise PluginError(f"[{self.name}] Plugin is in error state: {self.error_msg}")

        # Validate input
        validated = await self.validate(input_data)

        # Cache lookup
        if self.config.cache_results:
            cached = self._cache_get(validated)
            if cached is not None:
                logger.debug("[%s] Cache hit.", self.name)
                return cached

        async with self._lock:
            t0 = datetime.now(UTC)
            try:
                result = await asyncio.wait_for(
                    self.execute(validated, context),
                    timeout=self.config.timeout,
                )
                self._record_success(t0)

                if self.config.cache_results:
                    self._cache_set(validated, result)

                return result

            except asyncio.TimeoutError:
                self._record_failure(t0, f"timeout after {self.config.timeout}s")
                raise PluginTimeoutError(
                    f"[{self.name}] Timed out after {self.config.timeout}s"
                )
            except (PluginError, PluginValidationError):
                raise
            except Exception as exc:
                self._record_failure(t0, str(exc))
                raise PluginError(f"[{self.name}] Unexpected error: {exc}") from exc

    async def safe_execute_with_retry(
        self,
        input_data: Dict[str, Any],
        context: ExecutionContext,
    ) -> Any:
        """Execute with exponential-backoff retry up to max_retries."""
        last_exc: Optional[Exception] = None
        for attempt in range(self.config.max_retries + 1):
            try:
                return await self.safe_execute(input_data, context)
            except PluginError as exc:
                last_exc = exc
                if attempt < self.config.max_retries:
                    wait = 2 ** attempt
                    logger.warning(
                        "[%s] Retry %d/%d after %ds: %s",
                        self.name, attempt + 1, self.config.max_retries, wait, exc,
                    )
                    await asyncio.sleep(wait)
        raise last_exc or PluginError(f"[{self.name}] All retries exhausted.")

    # ─────────────────────────────────────────
    # Stats & health
    # ─────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        avg = self._time_total / self._exec_total if self._exec_total else 0.0
        success_rate = (
            self._exec_success / self._exec_total * 100 if self._exec_total else 0.0
        )
        return {
            "name":    self.name,
            "version": self.metadata.version,
            "status":  self.status.value,
            "enabled": self.config.enabled,
            "execution": {
                "total":        self._exec_total,
                "success":      self._exec_success,
                "failure":      self._exec_failure,
                "success_rate": f"{success_rate:.1f}%",
            },
            "timing": {
                "total_s":   round(self._time_total, 3),
                "avg_s":     round(avg, 3),
                "min_s":     round(self._time_min, 3) if self._time_min != float("inf") else None,
                "max_s":     round(self._time_max, 3),
                "last_s":    round(self._time_last, 3),
            },
            "timestamps": {
                "initialized":   self.initialized_at.isoformat()  if self.initialized_at  else None,
                "last_executed": self.last_executed_at.isoformat() if self.last_executed_at else None,
                "last_error":    self.last_error_at.isoformat()    if self.last_error_at    else None,
            },
            "error_journal_size": len(self._error_journal),
            "current_error":      self.error_msg,
        }

    def get_health(self) -> Dict[str, Any]:
        if self._exec_total == 0:
            label, score = "untested", 0
        else:
            rate = self._exec_success / self._exec_total
            if rate >= 0.95:
                label, score = "excellent", 100
            elif rate >= 0.80:
                label, score = "good", 80
            elif rate >= 0.60:
                label, score = "degraded", 60
            else:
                label, score = "critical", 30
        return {
            "label":   label,
            "score":   score,
            "status":  self.status.value,
            "enabled": self.config.enabled,
        }

    def reset_stats(self) -> None:
        self._exec_total   = 0
        self._exec_success = 0
        self._exec_failure = 0
        self._time_total   = 0.0
        self._time_min     = float("inf")
        self._time_max     = 0.0
        self._time_last    = 0.0
        self._error_journal.clear()

    # ─────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────

    def _record_success(self, t0: datetime) -> None:
        elapsed = (datetime.utcnow() - t0).total_seconds()
        self._exec_total   += 1
        self._exec_success += 1
        self._time_total   += elapsed
        self._time_last     = elapsed
        self._time_min      = min(self._time_min, elapsed)
        self._time_max      = max(self._time_max, elapsed)
        self.last_executed_at = datetime.utcnow()

    def _record_failure(self, t0: datetime, msg: str) -> None:
        elapsed = (datetime.utcnow() - t0).total_seconds()
        self._exec_total   += 1
        self._exec_failure += 1
        self._time_total   += elapsed
        self._time_last     = elapsed
        self.last_error_at  = datetime.utcnow()
        self.error_msg      = msg
        entry = {"ts": datetime.utcnow().isoformat(), "msg": msg, "elapsed_s": elapsed}
        self._error_journal.append(entry)
        if len(self._error_journal) > self._MAX_JOURNAL:
            self._error_journal.pop(0)
        if self._exec_failure > 10 and self._exec_failure / self._exec_total > 0.5:
            self.status = PluginStatus.ERROR

    def _cache_key(self, data: Dict[str, Any]) -> str:
        import hashlib, json
        return hashlib.sha256(
            json.dumps(data, sort_keys=True, default=str).encode()
        ).hexdigest()

    def _cache_get(self, data: Dict[str, Any]) -> Optional[Any]:
        import time
        key = self._cache_key(data)
        if key in self._result_cache:
            if time.time() - self._cache_ts[key] < self.config.cache_ttl_sec:
                return self._result_cache[key]
            del self._result_cache[key]
            del self._cache_ts[key]
        return None

    def _cache_set(self, data: Dict[str, Any], result: Any) -> None:
        import time
        key = self._cache_key(data)
        self._result_cache[key] = result
        self._cache_ts[key]     = time.time()

    # ─────────────────────────────────────────
    # Dunder
    # ─────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"version={self.metadata.version!r}, "
            f"status={self.status.value!r})"
        )
