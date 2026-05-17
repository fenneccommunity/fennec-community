"""
============
Abstract base class for all chains in the framework.

Design principles
-----------------
* `run()` is the **public API** — handles wrapping, timing, retries,
  fallbacks, tracing, and error handling.
* `_execute()` is the **internal hook** — subclasses implement this.
* `arun()` / `_aexecute()` provide async-first variants.
* The `>>` operator composes chains: `a >> b >> c`.

Example
-------
>>> class EchoChain(BaseChain):
...     async def _aexecute(self, inp: ChainInput) -> ChainOutput:
...         return ChainOutput(data=inp.data)
...
>>> chain = EchoChain()
>>> import asyncio
>>> asyncio.run(chain.arun("hello"))
'hello'
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from .config import ChainConfig
from .context import ChainInput, ChainOutput
from .tracing import ExecutionTracer

logger = logging.getLogger(__name__)


class BaseChain(ABC):
    """
    Abstract base for every chain type.

    Parameters
    ----------
    name   : Human-readable label (defaults to class name).
    config : Per-chain settings (retries, timeout, fallback, …).
    tracer : Shared ExecutionTracer; a new one is created if omitted.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        config: Optional[ChainConfig] = None,
        tracer: Optional[ExecutionTracer] = None,
    ) -> None:
        self.name: str = name or self.__class__.__name__
        self.chain_id: str = f"{self.name}-{str(uuid.uuid4())[:8]}"
        self.config: ChainConfig = config or ChainConfig()
        self.tracer: ExecutionTracer = tracer or ExecutionTracer()
        self._metrics: Dict[str, int] = defaultdict(int)
        self._history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Abstract interface — subclasses implement one or both
    # ------------------------------------------------------------------

    def _execute(self, input_data: ChainInput) -> ChainOutput:
        """
        Synchronous execution hook.

        Default implementation delegates to the async version via
        ``asyncio.run()``.  Override this if your chain is purely sync.
        """
        return asyncio.get_event_loop().run_until_complete(
            self._aexecute(input_data)
        )

    async def _aexecute(self, input_data: ChainInput) -> ChainOutput:
        """
        Async execution hook — override in subclasses.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _aexecute() or _execute()"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, input_data: Any, **kwargs: Any) -> Any:
        """
        Synchronous public entry point.

        Wraps input, handles retries, measures timing, records trace.
        """
        inp = self._wrap_input(input_data, kwargs)
        return asyncio.get_event_loop().run_until_complete(self._guarded_run(inp))

    async def arun(self, input_data: Any, **kwargs: Any) -> Any:
        """
        Async public entry point.

        Preferred over ``run()`` in async contexts.
        """
        inp = self._wrap_input(input_data, kwargs)
        return await self._guarded_run(inp)

    def __call__(self, input_data: Any, **kwargs: Any) -> Any:
        """Allow chains to be called like functions: chain(data)."""
        return self.run(input_data, **kwargs)

    # ------------------------------------------------------------------
    # Operator overloading for composition
    # ------------------------------------------------------------------

    def __rshift__(self, other: "BaseChain") -> "BaseChain":
        """
        Compose two chains: ``a >> b`` runs a then b.

        Returns a SequentialChain containing both.
        """
        # Lazy import to avoid circular dependency
        from ..chains import SequentialChain

        if isinstance(self, SequentialChain):
            # Flatten: avoid deeply nested SequentialChains
            return SequentialChain(self.chains + [other], tracer=self.tracer)
        return SequentialChain([self, other], tracer=self.tracer)

    def __or__(self, other: "BaseChain") -> "BaseChain":
        """Alias for >> (sequential composition)."""
        return self.__rshift__(other)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wrap_input(self, data: Any, kwargs: Dict[str, Any]) -> ChainInput:
        if isinstance(data, ChainInput):
            return data
        return ChainInput(data=data, metadata=kwargs)

    async def _guarded_run(self, inp: ChainInput) -> Any:
        """
        Execute with retry logic, timeout, tracing, and fallback.
        """
        attempts = self.config.retries + 1
        last_error: Optional[Exception] = None

        async with self._trace_span(inp) as span:
            for attempt in range(attempts):
                if attempt > 0:
                    await asyncio.sleep(self.config.retry_delay)
                    logger.debug(
                        "[%s] retry %d/%d", self.name, attempt, self.config.retries
                    )

                try:
                    output = await self._timed_execute(inp)

                    if output.success:
                        output.chain_id = self.chain_id
                        self._record(inp, output)
                        span.set_output(output.data)
                        return output.data

                    last_error = RuntimeError(output.error or "unknown error")

                except Exception as exc:
                    last_error = exc
                    logger.warning("[%s] attempt %d failed: %s", self.name, attempt + 1, exc)

            # All retries exhausted — try fallback
            if self.config.fallback_chain is not None:
                logger.info("[%s] invoking fallback chain", self.name)
                return await self.config.fallback_chain.arun(inp)

            # Hard failure
            self._metrics["error"] += 1
            raise last_error  # type: ignore[misc]

    async def _timed_execute(self, inp: ChainInput) -> ChainOutput:
        """Run _aexecute with optional timeout."""
        start = time.perf_counter()

        if self.config.timeout:
            coro = self._aexecute(inp)
            try:
                output = await asyncio.wait_for(coro, timeout=self.config.timeout)
            except asyncio.TimeoutError:
                elapsed = time.perf_counter() - start
                return ChainOutput(
                    data=None,
                    success=False,
                    error=f"Timeout after {elapsed:.2f}s (limit {self.config.timeout}s)",
                    execution_time=elapsed,
                )
        else:
            output = await self._aexecute(inp)

        output.execution_time = time.perf_counter() - start
        return output

    @asynccontextmanager
    async def _trace_span(self, inp: ChainInput):
        """Open a tracer span for this execution."""
        with self.tracer.span(self.name) as span:
            span.set_input(inp.data)
            yield span

    def _record(self, inp: ChainInput, output: ChainOutput) -> None:
        """Persist execution stats."""
        self._metrics["success"] += 1
        self._history.append(
            {
                "timestamp": time.time(),
                "input": str(inp.data)[:200],
                "output": str(output.data)[:200],
                "execution_time": output.execution_time,
                "success": output.success,
                "error": output.error,
            }
        )
        if self.config.verbose:
            logger.debug(
                "[%s] ✅ %.1fms  in=%s  out=%s",
                self.name,
                output.execution_time * 1000,
                str(inp.data)[:60],
                str(output.data)[:60],
            )

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return execution statistics for this chain."""
        total = self._metrics["success"] + self._metrics["error"]
        avg_time = (
            sum(h["execution_time"] for h in self._history) / len(self._history)
            if self._history
            else 0.0
        )
        return {
            "chain_id": self.chain_id,
            "name": self.name,
            "total_executions": total,
            "success_count": self._metrics["success"],
            "error_count": self._metrics["error"],
            "success_rate": self._metrics["success"] / total if total else 0.0,
            "avg_execution_time_ms": round(avg_time * 1000, 2),
        }

    def get_trace(self) -> Dict[str, Any]:
        """Return the last execution trace as a dict."""
        return self.tracer.to_dict()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, id={self.chain_id!r})"
