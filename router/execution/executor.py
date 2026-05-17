"""

The ExecutionEngine is responsible for *running* the chosen handler(s)
after the pipeline has selected candidates.

Supports
--------
- **Single** execution (best route only)
- **Sequential** top-k (execute in order, stop on first success)
- **Parallel** top-k (execute concurrently, aggregate results)
- **Retry** with configurable back-off
- **Streaming** via generator / async generator
- **Tool chaining** (sequential pipeline of route handlers)

All execution paths call :meth:`Route.execute` /
:meth:`Route.execute_async` so retry/metrics stay in one place.
"""
from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
from typing import Any, AsyncIterator, Callable, Iterator, List, Optional, Tuple

from ..config import ExecutionConfig, ExecutionMode
from ..core.base import HandlerRequest, HandlerResponse
from ..core.result import RouteCandidate, RoutingResult, RoutingTrace, MultiRoutingResult


# ---------------------------------------------------------------------------
# Retry Helper
# ---------------------------------------------------------------------------

def _with_retry(
    fn:         Callable[[], HandlerResponse],
    max_tries:  int,
    delay:      float,
    on_exceptions: List[str],
) -> HandlerResponse:
    """Run fn up to max_tries times, sleeping between attempts."""
    last_response: Optional[HandlerResponse] = None
    for attempt in range(max_tries):
        response = fn()
        if response.success:
            return response
        last_response = response
        # Only retry if the error class name is in the retry list
        if on_exceptions and response.error:
            should_retry = any(exc in (response.error or "") for exc in on_exceptions)
            if not should_retry:
                break
        if attempt < max_tries - 1:
            time.sleep(delay)
    return last_response  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Execution Engine
# ---------------------------------------------------------------------------

class ExecutionEngine:
    """
    Executes one or more route candidates and returns a
    :class:`RoutingResult` or :class:`MultiRoutingResult`.

    Parameters
    ----------
    cfg         : Execution configuration (mode, retries, timeout, …).
    route_map   : Dict mapping route name → Route instance.
                  Kept separate from candidates so the engine doesn't
                  need to reconstruct routes.
    aggregator  : Aggregation strategy for multi-route results.
    """

    def __init__(
        self,
        cfg:        ExecutionConfig,
        route_map:  dict,            # {name: Route}
        aggregator: "BaseAggregator | None" = None,
    ):
        self._cfg        = cfg
        self._route_map  = route_map
        self._aggregator = aggregator

    # ------------------------------------------------------------------ #
    # Sync Execution
    # ------------------------------------------------------------------ #

    def execute(
        self,
        request:    HandlerRequest,
        candidates: List[RouteCandidate],
        trace:      RoutingTrace,
    ) -> RoutingResult:
        """Dispatch based on configured ExecutionMode."""

        if not candidates:
            return RoutingResult(
                response = HandlerResponse.fail("No candidates available."),
                trace    = trace,
                matched  = False,
            )

        mode = self._cfg.mode

        if mode == ExecutionMode.SINGLE:
            return self._execute_single(request, candidates[0], trace)

        if mode == ExecutionMode.SEQUENTIAL:
            return self._execute_sequential(request, candidates, trace)

        if mode == ExecutionMode.PARALLEL:
            multi = self._execute_parallel(request, candidates, trace)
            # Unwrap into a RoutingResult for the common return type
            return RoutingResult(
                response = multi.aggregated,
                trace    = trace,
                matched  = multi.success,
            )

        raise ValueError(f"Unknown execution mode: {mode}")

    def _execute_single(
        self,
        request:   HandlerRequest,
        candidate: RouteCandidate,
        trace:     RoutingTrace,
    ) -> RoutingResult:
        route = self._route_map.get(candidate.route_name)
        if not route:
            return RoutingResult(
                response = HandlerResponse.fail(f"Route '{candidate.route_name}' not found in registry."),
                trace    = trace,
                matched  = False,
            )

        t_start = time.perf_counter()
        response = _with_retry(
            fn            = lambda: route.execute(request, candidate.combined_score),
            max_tries     = self._cfg.max_retries,
            delay         = self._cfg.retry_delay,
            on_exceptions = self._cfg.retry_on_exceptions,
        )
        trace.execution_time_ms = (time.perf_counter() - t_start) * 1000

        return RoutingResult(
            response = response,
            trace    = trace,
            matched  = response.success,
        )

    def _execute_sequential(
        self,
        request:    HandlerRequest,
        candidates: List[RouteCandidate],
        trace:      RoutingTrace,
    ) -> RoutingResult:
        """Try candidates in ranked order; return on first success."""
        t_start = time.perf_counter()
        for candidate in candidates:
            route = self._route_map.get(candidate.route_name)
            if not route:
                continue

            response = _with_retry(
                fn            = lambda r=route, c=candidate: r.execute(request, c.combined_score),
                max_tries     = self._cfg.max_retries,
                delay         = self._cfg.retry_delay,
                on_exceptions = self._cfg.retry_on_exceptions,
            )

            if response.success:
                trace.execution_time_ms = (time.perf_counter() - t_start) * 1000
                trace.selected_route    = candidate.route_name
                trace.selected_group    = candidate.group_name
                trace.add_note(f"Sequential: succeeded on '{candidate.route_name}'.")
                return RoutingResult(response=response, trace=trace, matched=True)

            trace.add_note(f"Sequential: '{candidate.route_name}' failed — trying next.")

        trace.execution_time_ms = (time.perf_counter() - t_start) * 1000
        return RoutingResult(
            response = HandlerResponse.fail("All sequential candidates failed."),
            trace    = trace,
            matched  = False,
        )

    def _execute_parallel(
        self,
        request:    HandlerRequest,
        candidates: List[RouteCandidate],
        trace:      RoutingTrace,
    ) -> MultiRoutingResult:
        """Execute all candidates concurrently; aggregate results."""
        t_start   = time.perf_counter()
        results: List[RoutingResult] = []

        with ThreadPoolExecutor(max_workers=len(candidates)) as pool:
            futures = {}
            for candidate in candidates:
                route = self._route_map.get(candidate.route_name)
                if not route:
                    continue
                fut = pool.submit(
                    _with_retry,
                    lambda r=route, c=candidate: r.execute(request, c.combined_score),
                    self._cfg.max_retries,
                    self._cfg.retry_delay,
                    self._cfg.retry_on_exceptions,
                )
                futures[fut] = candidate

            timeout = self._cfg.parallel_timeout
            try:
                for fut in as_completed(futures, timeout=timeout):
                    candidate  = futures[fut]
                    response   = fut.result()
                    results.append(RoutingResult(
                        response = response,
                        trace    = trace,
                        matched  = response.success,
                    ))
            except FuturesTimeout:
                trace.add_note(f"Parallel execution timed out after {timeout}s.")

        trace.execution_time_ms = (time.perf_counter() - t_start) * 1000

        aggregated = None
        if self._aggregator and results:
            aggregated = self._aggregator.aggregate(results)
        elif results:
            # Default: first successful response
            for r in results:
                if r.success:
                    aggregated = r.response
                    break
            if not aggregated:
                aggregated = results[0].response

        return MultiRoutingResult(
            results    = results,
            aggregated = aggregated,
            trace      = trace,
        )

    # ------------------------------------------------------------------ #
    # Async Execution
    # ------------------------------------------------------------------ #

    async def execute_async(
        self,
        request:    HandlerRequest,
        candidates: List[RouteCandidate],
        trace:      RoutingTrace,
    ) -> RoutingResult:
        """Async dispatch.  Parallel mode uses asyncio.gather."""
        if not candidates:
            return RoutingResult(
                response = HandlerResponse.fail("No candidates available."),
                trace    = trace,
                matched  = False,
            )

        mode = self._cfg.mode
        if mode == ExecutionMode.SINGLE:
            return await self._execute_single_async(request, candidates[0], trace)
        if mode == ExecutionMode.SEQUENTIAL:
            return await self._execute_sequential_async(request, candidates, trace)
        # Parallel
        multi = await self._execute_parallel_async(request, candidates, trace)
        return RoutingResult(
            response = multi.aggregated,
            trace    = trace,
            matched  = multi.success,
        )

    async def _execute_single_async(
        self,
        request:   HandlerRequest,
        candidate: RouteCandidate,
        trace:     RoutingTrace,
    ) -> RoutingResult:
        route = self._route_map.get(candidate.route_name)
        if not route:
            return RoutingResult(
                response=HandlerResponse.fail(f"Route '{candidate.route_name}' not found."),
                trace=trace, matched=False,
            )
        t0 = time.perf_counter()
        response = await route.execute_async(request, candidate.combined_score)
        trace.execution_time_ms = (time.perf_counter() - t0) * 1000
        return RoutingResult(response=response, trace=trace, matched=response.success)

    async def _execute_sequential_async(
        self,
        request:    HandlerRequest,
        candidates: List[RouteCandidate],
        trace:      RoutingTrace,
    ) -> RoutingResult:
        t0 = time.perf_counter()
        for candidate in candidates:
            route = self._route_map.get(candidate.route_name)
            if not route:
                continue
            response = await route.execute_async(request, candidate.combined_score)
            if response.success:
                trace.execution_time_ms = (time.perf_counter() - t0) * 1000
                trace.selected_route = candidate.route_name
                return RoutingResult(response=response, trace=trace, matched=True)
        trace.execution_time_ms = (time.perf_counter() - t0) * 1000
        return RoutingResult(
            response=HandlerResponse.fail("All async sequential candidates failed."),
            trace=trace, matched=False,
        )

    async def _execute_parallel_async(
        self,
        request:    HandlerRequest,
        candidates: List[RouteCandidate],
        trace:      RoutingTrace,
    ) -> MultiRoutingResult:
        t0 = time.perf_counter()
        tasks = []
        valid_candidates = []
        for c in candidates:
            route = self._route_map.get(c.route_name)
            if route:
                tasks.append(route.execute_async(request, c.combined_score))
                valid_candidates.append(c)

        try:
            responses = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self._cfg.parallel_timeout,
            )
        except asyncio.TimeoutError:
            responses = []
            trace.add_note("Async parallel timed out.")

        results = []
        for candidate, resp in zip(valid_candidates, responses):
            if isinstance(resp, Exception):
                resp = HandlerResponse.fail(str(resp))
            results.append(RoutingResult(response=resp, trace=trace, matched=resp.success))

        trace.execution_time_ms = (time.perf_counter() - t0) * 1000

        aggregated = None
        if results:
            for r in results:
                if r.success:
                    aggregated = r.response
                    break
            if not aggregated:
                aggregated = results[0].response if results else None

        return MultiRoutingResult(results=results, aggregated=aggregated, trace=trace)

    # ------------------------------------------------------------------ #
    # Streaming
    # ------------------------------------------------------------------ #

    def stream(
        self,
        request:    HandlerRequest,
        candidates: List[RouteCandidate],
    ) -> Iterator[str]:
        """Stream tokens from the best-matching route."""
        if not candidates:
            yield "[ERROR] No route candidates found."
            return

        route = self._route_map.get(candidates[0].route_name)
        if not route:
            yield "[ERROR] Route not found."
            return

        yield from route.handler.stream(request)

    async def stream_async(
        self,
        request:    HandlerRequest,
        candidates: List[RouteCandidate],
    ) -> AsyncIterator[str]:
        """Async streaming from the best-matching route."""
        if not candidates:
            yield "[ERROR] No route candidates found."
            return

        route = self._route_map.get(candidates[0].route_name)
        if not route:
            yield "[ERROR] Route not found."
            return

        async for chunk in route.handler.stream_async(request):
            yield chunk

    # ------------------------------------------------------------------ #
    # Tool Chaining
    # ------------------------------------------------------------------ #

    def chain(
        self,
        request:    HandlerRequest,
        route_names: List[str],
    ) -> HandlerResponse:
        """
        Execute a fixed sequence of routes as a pipeline.
        Each route's response content is passed into the next route's context.

        Example::

            result = engine.chain(request, ["extract_intent", "retrieve_docs", "generate_answer"])
        """
        context = dict(request.context)
        last_response: Optional[HandlerResponse] = None

        for name in route_names:
            route = self._route_map.get(name)
            if not route:
                return HandlerResponse.fail(f"Chain broken: route '{name}' not found.")

            chained_request = HandlerRequest(
                query      = request.query,
                context    = context,
                request_id = request.request_id,
                metadata   = request.metadata,
            )
            response = route.execute(chained_request)
            if not response.success:
                return HandlerResponse.fail(
                    f"Chain broken at '{name}': {response.error}"
                )

            # Pass result into context for next step
            context[f"step_{name}"] = response.content
            context["last_result"]  = response.content
            last_response = response

        return last_response or HandlerResponse.fail("Empty chain.")


# ---------------------------------------------------------------------------
# Base Aggregator
# ---------------------------------------------------------------------------

class BaseAggregator:
    """Abstract aggregation strategy for multi-route results."""
    def aggregate(self, results: List[RoutingResult]) -> HandlerResponse:
        raise NotImplementedError


class FirstWinsAggregator(BaseAggregator):
    """Return the first successful response."""
    def aggregate(self, results: List[RoutingResult]) -> HandlerResponse:
        for r in results:
            if r.success:
                return r.response  # type: ignore[return-value]
        return HandlerResponse.fail("No successful responses from any candidate.")


class MergeAggregator(BaseAggregator):
    """Merge all successful responses into a list payload."""
    def aggregate(self, results: List[RoutingResult]) -> HandlerResponse:
        merged = [r.content for r in results if r.success]
        if not merged:
            return HandlerResponse.fail("No successful responses to merge.")
        return HandlerResponse.ok(merged)
