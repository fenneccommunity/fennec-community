"""

HierarchicalRouter — the public entry-point for the entire system.

Architecture
------------

    ┌─────────────────────────────────────────────────────────────────┐
    │                     HierarchicalRouter                           │
    │                                                                  │
    │  ┌─────────────┐   ┌──────────────────────────────────────────┐ │
    │  │ CacheManager │   │           RoutingPipeline                │ │
    │  └─────────────┘   │  ┌────────────────────────────────────┐  │ │
    │                    │  │           HybridScorer              │  │ │
    │  ┌─────────────┐   │  │  SemanticScorer KeywordScorer LLM  │  │ │
    │  │RouterLogger  │   │  └────────────────────────────────────┘  │ │
    │  └─────────────┘   │  ┌────────────────────────────────────┐  │ │
    │                    │  │        ConfidenceEvaluator          │  │ │
    │  ┌─────────────┐   │  └────────────────────────────────────┘  │ │
    │  │MetricsCollect│  └──────────────────────────────────────────┘ │
    │  └─────────────┘                                                 │
    │                    ┌──────────────────────────────────────────┐ │
    │  ┌─────────────┐   │           ExecutionEngine                 │ │
    │  │FeedbackEngine│  │  single / sequential / parallel / stream │ │
    │  └─────────────┘   │  retry  |  tool chaining  |  async      │ │
    │                    └──────────────────────────────────────────┘ │
    │                                                                  │
    │  RouteGroups:  [RAG Group] [Tools Group] [Chat Group] …         │
    │  Routes:       Route objects stored in each group        │
    └─────────────────────────────────────────────────────────────────┘

Flow for a single query
-----------------------
  1. Normalise + cache lookup  (CacheManager)
  2. Embed query               (EmbeddingProvider)
  3. Score groups & routes     (RoutingPipeline → HybridScorer)
  4. Evaluate confidence       (ConfidenceEvaluator)
  5. Execute handler(s)        (ExecutionEngine)
  6. Record feedback           (FeedbackEngine)
  7. Record metrics            (MetricsCollector)
  8. Return RoutingResult
"""
from __future__ import annotations

import time
import uuid
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, Union

from ..cache.manager import CacheManager
from ..config import  RouterConfig
from ..core.base import BaseHandler, BaseRouter, CallableHandler, HandlerRequest, HandlerResponse
from ..core.result import RoutingResult, RoutingTrace
from ..core.route import Route
from ..core.route_group import RouteGroup
from ..execution.executor import ExecutionEngine, FirstWinsAggregator, MergeAggregator
from ..feedback.engine import FeedbackEngine
from ..observability.observability import MetricsCollector, RouterLogger
from ..routing.pipeline import ConfidenceEvaluator, EmbeddingProvider, RoutingPipeline
from ..routing.scorers import HybridScorer, KeywordScorer, LLMScorer, SemanticScorer


class HierarchicalRouter(BaseRouter):
    """
    Production-grade, extensible LLM routing system.

    Quick start
    -----------
    ::

        from fennec_community.router import HierarchicalRouter, RouteGroup, Route
        from fennec_community.router import BaseHandler, HandlerRequest, HandlerResponse

        class GreetHandler(BaseHandler):
            def handle(self, request):
                return HandlerResponse.ok(f"Hello! You said: {request.query}")

        router = HierarchicalRouter()

        chat = RouteGroup("chat", "General conversation")
        chat.add_route(Route(
            name="greet",
            description="Greet the user",
            handler=GreetHandler(),
            examples=["hello", "hi there", "good morning"],
        ))
        router.register_group(chat)

        result = router.route_query("Hey, how are you?")
        print(result.content)

    Parameters
    ----------
    config : RouterConfig  — full configuration object.
    """

    def __init__(self, config: Optional[RouterConfig] = None):
        self._cfg = config or RouterConfig()

        # ---- Observability -----------------------------------------------
        self._logger  = RouterLogger(self._cfg.observability)
        self._metrics = MetricsCollector(self._cfg.observability)

        # ---- Embedding model --------------------------------------------
        self._logger.info("Initialising embedding model", model=self._cfg.embedding.model_name)
        self._embedder = EmbeddingProvider(
            model_name = self._cfg.embedding.model_name,
            batch_size = self._cfg.embedding.batch_size,
        )

        # ---- Scorers ----------------------------------------------------
        llm_scorer: Optional[LLMScorer] = None
        if self._cfg.llm_model:
            llm_scorer = LLMScorer(
                model   = self._cfg.llm_model,
                api_key = self._cfg.llm_api_key,
            )

        self._scorer = HybridScorer(
            cfg      = self._cfg.scoring,
            semantic = SemanticScorer(self._cfg.scoring),
            keyword  = KeywordScorer(),
            llm      = llm_scorer,
        )

        # ---- Pipeline ---------------------------------------------------
        self._confidence = ConfidenceEvaluator(self._cfg.scoring)
        self._pipeline   = RoutingPipeline(
            cfg        = self._cfg,
            embedder   = self._embedder,
            scorer     = self._scorer,
            confidence = self._confidence,
        )

        # ---- Cache ------------------------------------------------------
        self._cache = CacheManager(self._cfg.cache)

        # ---- Feedback ---------------------------------------------------
        self._feedback = FeedbackEngine(self._cfg.feedback)

        # ---- Route Groups & flat route map ------------------------------
        self._groups:    Dict[str, RouteGroup]    = {}
        self._route_map: Dict[str, Route] = {}   # name → route

        # ---- Execution --------------------------------------------------
        aggregator = (
            MergeAggregator() if self._cfg.execution.aggregation.value == "merge"
            else FirstWinsAggregator()
        )
        self._executor = ExecutionEngine(
            cfg        = self._cfg.execution,
            route_map  = self._route_map,   # live reference — updates automatically
            aggregator = aggregator,
        )

        # ---- Global fallback --------------------------------------------
        self._fallback_handler: Optional[BaseHandler] = None

        self._logger.info("HierarchicalRouter ready")

    # ================================================================== #
    # Registration API
    # ================================================================== #

    def register_group(self, group: RouteGroup) -> "HierarchicalRouter":
        """
        Register a RouteGroup.

        All routes within the group are encoded immediately if they have
        examples and don't already have embeddings.

        Returns self for fluent chaining.
        """
        if group.name in self._groups:
            raise ValueError(
                f"Group '{group.name}' already registered. "
                f"Use update_group() to replace it."
            )

        # Encode group intent examples
        if group.intent_examples and not group.embeddings:
            self._embedder.encode_group(group)

        # Encode each route's examples
        for route in group:
            self._encode_route_if_needed(route)
            self._route_map[route.name] = route

        self._groups[group.name] = group
        self._logger.info(
            "Group registered",
            group  = group.name,
            routes = len(group),
        )
        return self

    def update_group(self, group: RouteGroup) -> "HierarchicalRouter":
        """Replace an existing group (or add if not present)."""
        # Remove old routes from flat map
        old = self._groups.get(group.name)
        if old:
            for route in old:
                self._route_map.pop(route.name, None)

        self._groups[group.name] = group
        if group.intent_examples:
            self._embedder.encode_group(group)
        for route in group:
            self._encode_route_if_needed(route)
            self._route_map[route.name] = route

        self._cache.clear()
        self._logger.info("Group updated", group=group.name)
        return self

    def register(self, route: Route, group_name: Optional[str] = None) -> None:
        """
        Register a single route.

        If ``group_name`` is given, the route is added to that group;
        otherwise it is placed in an auto-created ``"default"`` group.
        """
        target_group_name = group_name or "default"
        if target_group_name not in self._groups:
            self._groups[target_group_name] = RouteGroup(
                name        = target_group_name,
                description = "Default route group",
            )

        group = self._groups[target_group_name]
        group.add_route(route)
        self._encode_route_if_needed(route)
        self._route_map[route.name] = route
        self._cache.clear()

    def unregister(self, route_name: str) -> bool:
        """Remove a route by name."""
        route = self._route_map.pop(route_name, None)
        if route and route.group and route.group in self._groups:
            self._groups[route.group].remove_route(route_name)
        self._cache.clear()
        return route is not None

    def unregister_group(self, group_name: str) -> bool:
        """Remove an entire group and all its routes."""
        group = self._groups.pop(group_name, None)
        if not group:
            return False
        for route in group:
            self._route_map.pop(route.name, None)
        self._cache.clear()
        self._logger.info("Group unregistered", group=group_name)
        return True

    def set_fallback(self, handler: Union[BaseHandler, Callable]) -> None:
        """
        Set a global fallback handler invoked when no route matches.
        Accepts a BaseHandler instance or a plain callable.
        """
        self._fallback_handler = (
            handler if isinstance(handler, BaseHandler)
            else CallableHandler(handler)
        )
        self._logger.info("Fallback handler registered")

    # ================================================================== #
    # Primary Routing API
    # ================================================================== #

    def route_query(
        self,
        query:          str,
        context:        Optional[Dict[str, Any]] = None,
        return_result:  bool                     = True,
        **handler_kwargs,
    ) -> RoutingResult:
        """
        Route a query synchronously and return a :class:`RoutingResult`.

        Parameters
        ----------
        query          : The user's query string.
        context        : Optional context dict passed to the handler.
        return_result  : Always True — kept for backward compatibility.
        **handler_kwargs : Merged into request context.

        Returns
        -------
        RoutingResult with ``.content``, ``.success``, ``.trace``.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        request_id = str(uuid.uuid4())
        start_time = time.perf_counter()

        ctx = {**(context or {}), **handler_kwargs}
        request = HandlerRequest(
            query      = query,
            context    = ctx,
            request_id = request_id,
        )

        # ---- Cache lookup -----------------------------------------------
        cached = self._cache.get_decision(query)
        if cached:
            route_name, group_name = cached
            route = self._route_map.get(route_name)
            if route and route.enabled:
                self._logger.log_cache_hit(request_id, route_name)
                response  = route.execute(request)
                total_ms  = (time.perf_counter() - start_time) * 1000
                trace     = RoutingTrace(
                    request_id    = request_id,
                    query         = query,
                    selected_route = route_name,
                    selected_group = group_name,
                    used_cache    = True,
                    total_time_ms = total_ms,
                )
                result = RoutingResult(response=response, trace=trace, matched=True)
                self._post_route(result, request_id, route_name, group_name)
                return result

        # ---- Pipeline ---------------------------------------------------
        groups     = list(self._groups.values())
        candidates, trace = self._pipeline.run(query, groups, request_id)

        trace.total_time_ms = (time.perf_counter() - start_time) * 1000

        # ---- No match / low confidence ----------------------------------
        best_candidate = candidates[0] if candidates else None
        if not best_candidate or not self._confidence.should_route(best_candidate.confidence):
            return self._handle_no_match(request, trace, start_time)

        # ---- Cache the decision ------------------------------------------
        self._cache.set_decision(
            query,
            route_name = best_candidate.route_name,
            group_name = best_candidate.group_name,
        )

        # ---- Execute ----------------------------------------------------
        result = self._executor.execute(request, candidates, trace)
        result.trace.total_time_ms = (time.perf_counter() - start_time) * 1000

        # ---- Post-route hooks -------------------------------------------
        self._post_route(
            result, request_id,
            best_candidate.route_name,
            best_candidate.group_name,
        )
        return result

    # Alias
    def route(self, request: HandlerRequest) -> RoutingResult:
        """BaseRouter interface — wraps route_query."""
        return self.route_query(request.query, context=request.context)

    async def route_async(self, request: HandlerRequest) -> RoutingResult:
        """Async routing (non-blocking I/O via async handlers)."""
        query      = request.query
        request_id = request.request_id or str(uuid.uuid4())
        start_time = time.perf_counter()

        cached = self._cache.get_decision(query)
        if cached:
            route_name, group_name = cached
            route = self._route_map.get(route_name)
            if route and route.enabled:
                response = await route.execute_async(request)
                total_ms = (time.perf_counter() - start_time) * 1000
                trace    = RoutingTrace(
                    request_id     = request_id,
                    query          = query,
                    selected_route = route_name,
                    selected_group = group_name,
                    used_cache     = True,
                    total_time_ms  = total_ms,
                )
                return RoutingResult(response=response, trace=trace, matched=True)

        groups = list(self._groups.values())
        candidates, trace = self._pipeline.run(query, groups, request_id)
        trace.total_time_ms = (time.perf_counter() - start_time) * 1000

        best = candidates[0] if candidates else None
        if not best or not self._confidence.should_route(best.confidence):
            return self._handle_no_match(request, trace, start_time)

        self._cache.set_decision(query, best.route_name, best.group_name)
        result = await self._executor.execute_async(request, candidates, trace)
        result.trace.total_time_ms = (time.perf_counter() - start_time) * 1000
        self._post_route(result, request_id, best.route_name, best.group_name)
        return result

    def stream(
        self,
        query:   str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Iterator[str]:
        """Stream tokens from the best-matching handler."""
        request    = HandlerRequest(query=query, context=context or {})
        groups     = list(self._groups.values())
        candidates, _ = self._pipeline.run(query, groups, request.request_id)
        yield from self._executor.stream(request, candidates)

    async def stream_async(
        self,
        query:   str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[str]:
        """Async token streaming from the best-matching handler."""
        request    = HandlerRequest(query=query, context=context or {})
        groups     = list(self._groups.values())
        candidates, _ = self._pipeline.run(query, groups, request.request_id)
        async for chunk in self._executor.stream_async(request, candidates):
            yield chunk

    def chain(
        self,
        query:       str,
        route_names: List[str],
        context:     Optional[Dict[str, Any]] = None,
    ) -> HandlerResponse:
        """Execute a fixed chain of routes in sequence."""
        request = HandlerRequest(query=query, context=context or {})
        return self._executor.chain(request, route_names)

    # ================================================================== #
    # Feedback
    # ================================================================== #

    def feedback(
        self,
        route_name: str,
        success:    bool,
        score:      float  = 0.0,
        query:      str    = "",
    ) -> None:
        """
        Record a feedback signal for a routing outcome.

        Call this after the application has determined whether the
        routed response was actually useful (e.g. user clicked thumbs up).
        """
        group_name = self._route_map[route_name].group if route_name in self._route_map else None
        self._feedback.record(route_name, group_name, success, score, query)
        self._feedback.apply_biases(self._route_map)
        self._logger.log_feedback(route_name, success, score)

    # ================================================================== #
    # Re-encoding
    # ================================================================== #

    def re_encode_route(self, route_name: str) -> bool:
        """Force re-computation of a route's embeddings (after adding examples)."""
        route = self._route_map.get(route_name)
        if not route:
            return False
        self._embedder.encode_route(route)
        self._cache.clear()
        return True

    def re_encode_group(self, group_name: str) -> bool:
        """Re-encode all routes in a group."""
        group = self._groups.get(group_name)
        if not group:
            return False
        if group.intent_examples:
            self._embedder.encode_group(group)
        for route in group:
            self._embedder.encode_route(route)
        self._cache.clear()
        return True

    # ================================================================== #
    # Observability & Inspection
    # ================================================================== #

    def metrics(self) -> Dict[str, Any]:
        return self._metrics.snapshot()

    def cache_stats(self) -> Dict[str, Any]:
        return self._cache.stats()

    def feedback_summary(self) -> Dict[str, Any]:
        return self._feedback.summary()

    def list_groups(self) -> List[str]:
        return list(self._groups.keys())

    def list_routes(self, group_name: Optional[str] = None) -> List[str]:
        if group_name:
            group = self._groups.get(group_name)
            return [r.name for r in group] if group else []
        return list(self._route_map.keys())

    def get_route(self, name: str) -> Optional[Route]:
        return self._route_map.get(name)

    def get_group(self, name: str) -> Optional[RouteGroup]:
        return self._groups.get(name)

    def summary(self) -> Dict[str, Any]:
        return {
            "groups":       {n: g.summary() for n, g in self._groups.items()},
            "total_routes": len(self._route_map),
            "config": {
                "model":                self._cfg.embedding.model_name,
                "execution_mode":       self._cfg.execution.mode.value,
                "top_k":                self._cfg.scoring.top_k,
                "high_threshold":       self._cfg.scoring.high_confidence_threshold,
                "medium_threshold":     self._cfg.scoring.medium_confidence_threshold,
                "fallback_threshold":   self._cfg.scoring.fallback_confidence_threshold,
                "cache_enabled":        self._cfg.cache.enabled,
                "feedback_enabled":     self._cfg.feedback.enabled,
            },
            "metrics": self.metrics(),
        }

    # ================================================================== #
    # Context Manager
    # ================================================================== #

    def __enter__(self) -> "HierarchicalRouter":
        return self

    def __exit__(self, *_):
        if self._cfg.feedback.persist_path:
            self._feedback.save()

    def __call__(self, query: str, **kwargs) -> RoutingResult:
        return self.route_query(query, **kwargs)

    # ================================================================== #
    # Private Helpers
    # ================================================================== #

    def _encode_route_if_needed(self, route: Route):
        if route.examples and not route.embeddings:
            self._embedder.encode_route(route)

    def _handle_no_match(
        self,
        request:    HandlerRequest,
        trace:      RoutingTrace,
        start_time: float,
    ) -> RoutingResult:
        top_score = (
            trace.candidates[0].effective_score if trace.candidates else 0.0
        )
        self._logger.log_no_match(request.query, trace.request_id, top_score)

        if self._fallback_handler:
            response = self._fallback_handler.handle(request)
            trace.total_time_ms = (time.perf_counter() - start_time) * 1000
            trace.add_note("Fallback handler invoked.")
            result = RoutingResult(response=response, trace=trace, matched=False)
            self._metrics.record(result)
            return result

        if self._cfg.raise_on_no_match:
            raise LookupError(
                f"No route matched query (top score: {top_score:.3f}). "
                f"Consider lowering fallback_confidence_threshold or adding examples."
            )

        response = HandlerResponse.fail(
            f"No route matched. Top score: {top_score:.3f}. "
            f"Threshold: {self._cfg.scoring.fallback_confidence_threshold}."
        )
        trace.total_time_ms = (time.perf_counter() - start_time) * 1000
        result = RoutingResult(response=response, trace=trace, matched=False)
        self._metrics.record(result)
        return result

    def _post_route(
        self,
        result:     RoutingResult,
        request_id: str,
        route_name: str,
        group_name: Optional[str],
    ):
        """Record metrics + auto-feedback after a routed call."""
        result.trace.selected_route = route_name
        result.trace.selected_group = group_name
        self._metrics.record(result)
        self._logger.log_routing_decision(result)

        # Auto-feedback based on handler success flag
        if self._cfg.feedback.enabled:
            score = (
                result.trace.candidates[0].effective_score
                if result.trace.candidates else 0.0
            )
            self._feedback.record(
                route_name = route_name,
                group_name = group_name,
                success    = result.success,
                score      = score,
            )
            self._feedback.apply_biases(self._route_map)
