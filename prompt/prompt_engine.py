"""
Prompt Engine — Master Orchestrator
=====================================
The single public entry point for the entire prompt system.

  engine = PromptEngine()

  prompt = engine.build(
      query     = "What caused the 2008 financial crisis?",
      documents = retrieved_docs,
      memory    = chat_history,
      strategy  = "multi_hop",
  )

  # Feed directly into your LLM client
  response = llm.chat(prompt.to_messages())

Key responsibilities
--------------------
* Accept flexible inputs (raw strings or typed objects)
* Auto-detect prompt type and optimal strategy when not provided
* Coordinate builder → context manager → guardrails → optimizer
* Emit structured observability events (metrics, traces)
* Adaptive feedback loop: record quality signals for future prompt improvement
* Integration hooks for router, memory store, cache
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from .builder import PromptBuilder
from .context_manager import ContextManager
from .guardrails import Guardrail, GuardrailEngine
from .optimizer import PromptOptimizer
from .types import (
    BuiltPrompt,
    Document,
    Message,
    OutputFormat,
    PromptRequest,
    PromptStrategy,
    PromptType,
    QueryComplexity,
    UserProfile,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Observability
# ─────────────────────────────────────────────

@dataclass
class PromptMetrics:
    total_builds:      int   = 0
    total_tokens:      int   = 0
    total_tokens_saved: int  = 0
    cache_hits:        int   = 0
    builds_by_type:    Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    builds_by_strategy: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    avg_build_ms:      float = 0.0
    _total_build_ms:   float = field(default=0.0, repr=False)

    def record(self, prompt: BuiltPrompt, elapsed_ms: float) -> None:
        self.total_builds      += 1
        self.total_tokens      += prompt.estimated_tokens
        self.total_tokens_saved += prompt.tokens_saved
        self.builds_by_type[prompt.prompt_type.value]     += 1
        self.builds_by_strategy[prompt.strategy.value]   += 1
        self._total_build_ms   += elapsed_ms
        self.avg_build_ms       = self._total_build_ms / self.total_builds

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_builds":        self.total_builds,
            "total_tokens":        self.total_tokens,
            "total_tokens_saved":  self.total_tokens_saved,
            "cache_hits":          self.cache_hits,
            "avg_build_ms":        round(self.avg_build_ms, 2),
            "cache_hit_rate_pct":  (
                round(self.cache_hits / self.total_builds * 100, 1)
                if self.total_builds else 0
            ),
            "builds_by_type":      dict(self.builds_by_type),
            "builds_by_strategy":  dict(self.builds_by_strategy),
        }


# ─────────────────────────────────────────────
# Adaptive feedback entry
# ─────────────────────────────────────────────

@dataclass
class FeedbackEntry:
    trace_id:      str
    prompt_type:   str
    strategy:      str
    quality_score: float    # 0.0 = bad, 1.0 = perfect
    notes:         str = ""


# ─────────────────────────────────────────────
# Prompt Engine
# ─────────────────────────────────────────────

class PromptEngine:
    """
    Production-grade prompt orchestration engine.

    Features
    --------
    • Auto-detection of prompt type, strategy, and complexity
    • Pluggable context manager, guardrail engine, optimizer
    • Built-in prompt caching (hash-based)
    • Adaptive prompting via feedback loop
    • Integration hooks for external RAG components
    • Full observability: metrics + per-prompt trace log
    """

    def __init__(
        self,
        # Subsystem overrides (optional — defaults are production-ready)
        context_manager:  Optional[ContextManager]  = None,
        guardrail_engine: Optional[GuardrailEngine] = None,
        extra_guardrails: Optional[List[Guardrail]]  = None,
        # Feature flags
        enable_cache:       bool  = True,
        cache_ttl_sec:      int   = 300,
        max_cache_size:     int   = 256,
        enable_auto_detect: bool  = True,   # auto-detect type/strategy
        # Integration handles (injected externally)
        memory_store:  Optional[Any] = None,
        cache_store:   Optional[Any] = None,
        router:        Optional[Any] = None,
    ) -> None:
        self.enable_cache       = enable_cache
        self.cache_ttl_sec      = cache_ttl_sec
        self.max_cache_size     = max_cache_size
        self.enable_auto_detect = enable_auto_detect
        self.memory_store       = memory_store
        self.cache_store        = cache_store
        self.router             = router

        # Build subsystems
        _guardrail_engine = guardrail_engine or GuardrailEngine(
            extra_guardrails=extra_guardrails
        )
        self._builder = PromptBuilder(
            context_manager  = context_manager or ContextManager(),
            guardrail_engine = _guardrail_engine,
        )

        # State
        self._metrics:  PromptMetrics             = PromptMetrics()
        self._cache:    Dict[str, tuple]          = {}   # hash → (BuiltPrompt, timestamp)
        self._feedback: List[FeedbackEntry]       = []
        self._hooks:    Dict[str, List[Callable]] = defaultdict(list)
        self._trace_log: List[Dict[str, Any]]    = []

        logger.info(
            "[PromptEngine] Initialized (cache=%s, auto_detect=%s)",
            enable_cache, enable_auto_detect,
        )

    # ─────────────────────────────────────────
    # Primary API
    # ─────────────────────────────────────────

    def build(
        self,
        # Core inputs (flexible types)
        query:      str,
        documents:  Optional[List[Union[Document, Dict, str]]] = None,
        memory:     Optional[List[Union[Message, Dict]]]        = None,
        # Prompt configuration
        prompt_type:    Union[PromptType,   str] = PromptType.QA,
        strategy:       Union[PromptStrategy, str] = PromptStrategy.SIMPLE,
        output_format:  Union[OutputFormat,  str] = OutputFormat.TEXT,
        complexity:     Union[QueryComplexity, str] = QueryComplexity.SIMPLE,
        user_profile:   Union[UserProfile,   str] = UserProfile.GENERAL,
        # Constraints
        max_context_tokens: int            = 3000,
        max_answer_tokens:  int            = 512,
        output_schema:      Optional[Dict] = None,
        language:           str            = "en",
        # Feature flags
        enable_guardrails:  bool           = True,
        enable_citations:   bool           = True,
        enable_uncertainty: bool           = True,
        # Metadata
        session_id: str = "",
        user_id:    str = "",
        trace_id:   str = "",
        extra:      Optional[Dict] = None,
    ) -> BuiltPrompt:
        """
        Build an optimised, context-aware prompt.

        Parameters
        ----------
        query           : user's question or task description
        documents       : retrieved passages (Document, dict, or plain string)
        memory          : conversation history (Message or dict)
        prompt_type     : type of prompt (qa, reasoning, agent, …)
        strategy        : building strategy (simple, cot, multi_hop, …)
        output_format   : desired response format (text, json, markdown, …)
        complexity      : query complexity (simple → expert)
        user_profile    : reader profile (general, technical, academic, executive)
        max_context_tokens : token budget for context injection
        max_answer_tokens  : hint to the LLM about expected answer length
        output_schema      : JSON schema for structured output
        language           : response language code (e.g. 'en', 'ar', 'fr')
        enable_guardrails  : inject anti-hallucination instructions
        enable_citations   : request inline source citations
        enable_uncertainty : ask model to express uncertainty honestly
        session_id, user_id, trace_id : tracing / logging identifiers

        Returns
        -------
        BuiltPrompt
        """
        t0 = time.monotonic()

        # Normalise inputs
        docs  = self._normalise_documents(documents or [])
        msgs  = self._normalise_messages(memory or [])
        ptype = PromptType(prompt_type)   if isinstance(prompt_type, str)   else prompt_type
        strat = PromptStrategy(strategy)  if isinstance(strategy,    str)   else strategy
        ofmt  = OutputFormat(output_format) if isinstance(output_format, str) else output_format
        cplx  = QueryComplexity(complexity) if isinstance(complexity, str)   else complexity
        uprof = UserProfile(user_profile)   if isinstance(user_profile, str) else user_profile

        # Auto-detect overrides
        if self.enable_auto_detect:
            ptype, strat, cplx = self._auto_detect(query, docs, ptype, strat, cplx)

        request = PromptRequest(
            query               = query,
            documents           = docs,
            memory              = msgs,
            prompt_type         = ptype,
            strategy            = strat,
            output_format       = ofmt,
            complexity          = cplx,
            user_profile        = uprof,
            max_context_tokens  = max_context_tokens,
            max_answer_tokens   = max_answer_tokens,
            output_schema       = output_schema,
            language            = language,
            enable_guardrails   = enable_guardrails,
            enable_citations    = enable_citations,
            enable_uncertainty  = enable_uncertainty,
            session_id          = session_id,
            user_id             = user_id,
            trace_id            = trace_id,
            extra               = extra or {},
        )

        # Cache check
        if self.enable_cache:
            cached = self._cache_get(request)
            if cached is not None:
                self._metrics.cache_hits += 1
                logger.debug("[PromptEngine] Cache hit for query: %.60s", query)
                return cached

        # Build
        prompt = self._builder.build(request)

        # Cache store
        if self.enable_cache:
            self._cache_set(request, prompt)

        # Observability
        elapsed_ms = (time.monotonic() - t0) * 1000
        self._metrics.record(prompt, elapsed_ms)
        self._log_trace(request, prompt, elapsed_ms)
        self._fire_hook("prompt.built", prompt=prompt, request=request)

        logger.info(
            "[PromptEngine] Built prompt | type=%s strategy=%s tokens=%d saved=%d ms=%.1f",
            prompt.prompt_type.value,
            prompt.strategy.value,
            prompt.estimated_tokens,
            prompt.tokens_saved,
            elapsed_ms,
        )
        return prompt

    def build_from_request(self, request: PromptRequest) -> BuiltPrompt:
        """Build directly from a PromptRequest object."""
        return self.build(
            query               = request.query,
            documents           = request.documents,
            memory              = request.memory,
            prompt_type         = request.prompt_type,
            strategy            = request.strategy,
            output_format       = request.output_format,
            complexity          = request.complexity,
            user_profile        = request.user_profile,
            max_context_tokens  = request.max_context_tokens,
            max_answer_tokens   = request.max_answer_tokens,
            output_schema       = request.output_schema,
            language            = request.language,
            enable_guardrails   = request.enable_guardrails,
            enable_citations    = request.enable_citations,
            enable_uncertainty  = request.enable_uncertainty,
            session_id          = request.session_id,
            user_id             = request.user_id,
            trace_id            = request.trace_id,
            extra               = request.extra,
        )

    # ─────────────────────────────────────────
    # Adaptive feedback
    # ─────────────────────────────────────────

    def record_feedback(
        self,
        trace_id:      str,
        quality_score: float,
        notes:         str = "",
    ) -> None:
        """
        Record a quality signal for a previously built prompt.
        Used to guide future strategy selection.
        """
        # Find the trace
        trace = next((t for t in self._trace_log if t["trace_id"] == trace_id), None)
        entry = FeedbackEntry(
            trace_id      = trace_id,
            prompt_type   = trace["prompt_type"]   if trace else "unknown",
            strategy      = trace["strategy"]       if trace else "unknown",
            quality_score = quality_score,
            notes         = notes,
        )
        self._feedback.append(entry)
        if len(self._feedback) > 1000:
            self._feedback.pop(0)
        logger.info(
            "[PromptEngine] Feedback recorded: trace=%s score=%.2f",
            trace_id, quality_score,
        )

    def adaptive_strategy_for(self, prompt_type: PromptType) -> Optional[PromptStrategy]:
        """
        Suggest the historically best-performing strategy for a prompt type,
        based on accumulated feedback. Returns None if insufficient data.
        """
        relevant = [
            f for f in self._feedback
            if f.prompt_type == prompt_type.value
        ]
        if len(relevant) < 5:
            return None

        from collections import defaultdict
        scores: Dict[str, List[float]] = defaultdict(list)
        for f in relevant:
            scores[f.strategy].append(f.quality_score)

        best = max(scores, key=lambda s: sum(scores[s]) / len(scores[s]))
        return PromptStrategy(best)

    # ─────────────────────────────────────────
    # Observability
    # ─────────────────────────────────────────

    @property
    def metrics(self) -> Dict[str, Any]:
        return self._metrics.to_dict()

    def get_trace_log(self, last_n: int = 20) -> List[Dict[str, Any]]:
        return self._trace_log[-last_n:]

    def reset_metrics(self) -> None:
        self._metrics = PromptMetrics()
        logger.info("[PromptEngine] Metrics reset.")

    # ─────────────────────────────────────────
    # Hook system
    # ─────────────────────────────────────────

    def on(self, event: str, callback: Callable) -> None:
        self._hooks[event].append(callback)

    def _fire_hook(self, event: str, **kwargs) -> None:
        for cb in self._hooks.get(event, []):
            try:
                cb(**kwargs)
            except Exception as exc:
                logger.warning("[PromptEngine] Hook '%s' error: %s", event, exc)

    # ─────────────────────────────────────────
    # Auto-detection
    # ─────────────────────────────────────────

    def _auto_detect(
        self,
        query:     str,
        documents: List[Document],
        ptype:     PromptType,
        strat:     PromptStrategy,
        cplx:      QueryComplexity,
    ):
        q = query.lower()

        # Prompt type inference
        if ptype == PromptType.QA:
            if any(w in q for w in ("summarize", "summarise", "summary", "overview")):
                ptype = PromptType.SUMMARIZATION
            elif any(w in q for w in ("compare", "contrast", "difference", "versus", "vs")):
                ptype = PromptType.COMPARISON
            elif any(w in q for w in ("extract", "list all", "find all", "enumerate")):
                ptype = PromptType.EXTRACTION
            elif any(w in q for w in ("why", "how", "explain", "reason", "cause")):
                ptype = PromptType.REASONING

        # Complexity inference
        word_count = len(query.split())
        if word_count > 40 or len(documents) > 5:
            cplx = QueryComplexity.COMPLEX
        elif word_count > 20 or len(documents) > 2:
            cplx = QueryComplexity.MODERATE

        multi_hop_signals = ("and", "also", "additionally", "furthermore", "relationship between")
        if any(sig in q for sig in multi_hop_signals) and len(documents) > 1:
            strat = PromptStrategy.MULTI_HOP

        return ptype, strat, cplx

    # ─────────────────────────────────────────
    # Input normalisation
    # ─────────────────────────────────────────

    @staticmethod
    def _normalise_documents(raw: List[Any]) -> List[Document]:
        out: List[Document] = []
        for item in raw:
            if isinstance(item, Document):
                out.append(item)
            elif isinstance(item, dict):
                out.append(Document(
                    content  = item.get("content", item.get("text", str(item))),
                    source   = item.get("source", item.get("url", "")),
                    score    = float(item.get("score", 1.0)),
                    metadata = item.get("metadata", {}),
                ))
            elif isinstance(item, str):
                out.append(Document(content=item))
        return out

    @staticmethod
    def _normalise_messages(raw: List[Any]) -> List[Message]:
        out: List[Message] = []
        for item in raw:
            if isinstance(item, Message):
                out.append(item)
            elif isinstance(item, dict):
                out.append(Message(
                    role    = item.get("role", "user"),
                    content = item.get("content", ""),
                ))
        return out

    # ─────────────────────────────────────────
    # Cache
    # ─────────────────────────────────────────

    def _cache_key(self, request: PromptRequest) -> str:
        payload = json.dumps({
            "query":       request.query,
            "doc_hashes":  [d.content[:100] for d in request.documents],
            "type":        request.prompt_type.value,
            "strategy":    request.strategy.value,
            "format":      request.output_format.value,
            "language":    request.language,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def _cache_get(self, request: PromptRequest) -> Optional[BuiltPrompt]:
        key = self._cache_key(request)
        entry = self._cache.get(key)
        if entry is None:
            return None
        prompt, ts = entry
        if time.time() - ts > self.cache_ttl_sec:
            del self._cache[key]
            return None
        return prompt

    def _cache_set(self, request: PromptRequest, prompt: BuiltPrompt) -> None:
        if len(self._cache) >= self.max_cache_size:
            # Evict oldest entry
            oldest = min(self._cache, key=lambda k: self._cache[k][1])
            del self._cache[oldest]
        self._cache[self._cache_key(request)] = (prompt, time.time())

    # ─────────────────────────────────────────
    # Trace log
    # ─────────────────────────────────────────

    def _log_trace(
        self,
        request:    PromptRequest,
        prompt:     BuiltPrompt,
        elapsed_ms: float,
    ) -> None:
        entry = {
            "trace_id":          request.trace_id or prompt.trace_id,
            "session_id":        request.session_id,
            "query_preview":     request.query[:80],
            "prompt_type":       prompt.prompt_type.value,
            "strategy":          prompt.strategy.value,
            "output_format":     prompt.output_format.value,
            "docs_included":     prompt.documents_included,
            "docs_truncated":    prompt.documents_truncated,
            "estimated_tokens":  prompt.estimated_tokens,
            "tokens_saved":      prompt.tokens_saved,
            "guardrails":        prompt.guardrails_applied,
            "elapsed_ms":        round(elapsed_ms, 2),
            "ts":                time.time(),
        }
        self._trace_log.append(entry)
        if len(self._trace_log) > 500:
            self._trace_log.pop(0)

    def __repr__(self) -> str:
        return (
            f"PromptEngine(builds={self._metrics.total_builds}, "
            f"cache_size={len(self._cache)}, "
            f"cache_hit_rate={self.metrics.get('cache_hit_rate_pct', 0)}%)"
        )
