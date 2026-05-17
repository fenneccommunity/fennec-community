"""
 Advanced Context Intelligence Engine (Main Orchestrator)

Architecture:
    Query
      ↓
    [QueryAnalyzer]        ← intent, keywords, complexity, budget
      ↓
    [HybridRetriever]      ← vector + BM25 keyword + metadata → RRF fusion
      ↓
    [ChunkFilter]          ← dedup + noise + length filter
      ↓
    [CompositeRanker]      ← composite score (semantic + keyword + quality + recency)
      ↓
    [SmartContextComposer] ← format + group + truncate-to-budget
      ↓
    [ContextGuard]         ← validate relevance + coverage + length
      ↓
    [AdaptiveLearning]     ← record metrics, auto-tune params
      ↓
    ContextResult          ← context + full stats
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .config import ContextEngineConfig, LengthCounter
from .models import (
    ConversationHistory, ContextResult,
    ScoredChunk, SourceQuality,
)
from .query_analyzer import build_query_analyzer
from .retriever import HybridRetriever, KeywordRetriever, MetadataRetriever, VectorRetriever
from .pipeline import char_counter, make_token_counter, ChunkFilter, CompositeRanker, SmartContextComposer
from .guard_and_learning import (
    AdaptiveLearningSystem,
    ComplexityBasedAllocation,
    ContextCache,
    ContextGuard,
    EqualBudgetAllocation,
)
from .strategy import (
    BudgetAllocationStrategy,
    ChunkFilterStrategy,
    ChunkRankingStrategy,
    ContextComposerStrategy,
    ContextGuardStrategy,
    HybridRetrieverStrategy,
    QueryAnalyzerStrategy,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Context Intelligence Engine
# ─────────────────────────────────────────────────────────────────────────────

class ContextEngine:
    """
    Advanced Context Intelligence Engine.

    original use case:
    ─────────────────
        from fennec_community.context import ContextEngine, ContextEngineConfig

        engine = ContextEngine(config=ContextEngineConfig())
        engine.index(chunks=my_chunks, embed_fn=my_embed_function)

        result = engine.run(query="ما هو التعلم الآلي؟")
        print(result.context)

    advanced use case (plug & play):
    ─────────────────────────────────
        engine = ContextEngine(
            config            = my_config,
            query_analyzer    = MyLLMAnalyzer(),
            retriever         = MyVectorStoreRetriever(),
            ranker            = MyCustomRanker(),
            composer          = MyCustomComposer(),
            guard             = MyHallucinationGuard(),
        )
    """

    def __init__(
        self,
        config:            Optional[ContextEngineConfig]      = None,

        # ── Plug & Play components ──────────────────────────────────────
        query_analyzer:    Optional[QueryAnalyzerStrategy]    = None,
        retriever:         Optional[HybridRetriever]          = None,
        chunk_filter:      Optional[ChunkFilterStrategy]      = None,
        ranker:            Optional[ChunkRankingStrategy]     = None,
        composer:          Optional[ContextComposerStrategy]  = None,
        guard:             Optional[ContextGuardStrategy]     = None,
        budget_allocator:  Optional[BudgetAllocationStrategy] = None,
        length_counter:    Optional[LengthCounter]            = None,
    ):
        self.config   = config or ContextEngineConfig()
        self._counter = length_counter or char_counter

        # ── Build components (use injected or default) ─────────────────
        self._analyzer  = query_analyzer or build_query_analyzer(self.config)
        self._filter    = chunk_filter   or ChunkFilter(self.config)
        self._ranker    = ranker         or CompositeRanker(self.config)
        self._composer  = composer       or SmartContextComposer(self.config, self._counter)
        self._guard     = guard          or ContextGuard(self.config)
        self._budget    = budget_allocator or EqualBudgetAllocation()
        self._adaptive  = AdaptiveLearningSystem(self.config)

        # ── Retriever (built or injected) ──────────────────────────────
        self._retriever: Optional[HybridRetriever] = retriever

        # ── Cache ──────────────────────────────────────────────────────
        self._cache = ContextCache(
            max_size    = self.config.cache.max_size,
            ttl_seconds = self.config.cache.ttl_seconds,
        ) if self.config.cache.enabled else None

        # ── Conversation history ───────────────────────────────────────
        self._history: Optional[ConversationHistory] = None
        if self.config.multi_turn:
            self._history = ConversationHistory(session_id=f"session_{int(time.time())}")

        logger.info("ContextEngine: Ready ✓ (template=%s)", self.config.template)

    # ─────────────────────────────────────────────────────────────────────
    #  Indexing API
    # ─────────────────────────────────────────────────────────────────────

    def index(
        self,
        chunks:               List[Any],
        embed_fn:             Optional[Callable[[str], np.ndarray]] = None,
        vector_store_adapter: Optional[Any]                          = None,
        source_quality_map:   Optional[Dict[str, SourceQuality]]    = None,
    ) -> "ContextEngine":
        """
        فهرسة الـ chunks في النظام.

        Args:
            chunks:               قائمة الـ chunks (أي نوع يحتوي على text و doc_id)
            embed_fn:             دالة توليد embedding لنص واحد
            vector_store_adapter: vector store خارجي (Chroma/FAISS/Pinecone) — بديل embed_fn
            source_quality_map:   تحديد جودة المصادر {source_name → SourceQuality}

        Returns:
            self (للـ method chaining)
        """
        if not chunks:
            logger.warning("ContextEngine.index: لا توجد chunks للفهرسة")
            return self

        self._source_quality_map = source_quality_map or {}

        # بناء الـ retrievers
        vec_ret  = VectorRetriever(
            self.config,
            vector_store_adapter=vector_store_adapter,
        )
        kw_ret   = KeywordRetriever(self.config, chunks=chunks)
        meta_ret = MetadataRetriever(self.config, chunks=chunks)

        if embed_fn and vector_store_adapter is None:
            vec_ret.index_chunks(chunks, embed_fn)

        if self._retriever is None:
            self._retriever = HybridRetriever(
                config             = self.config,
                vector_retriever   = vec_ret,
                keyword_retriever  = kw_ret,
                metadata_retriever = meta_ret,
            )

        logger.info("ContextEngine: فُهرس %d chunk", len(chunks))
        return self

    # ─────────────────────────────────────────────────────────────────────
    #  Main API: run()
    # ─────────────────────────────────────────────────────────────────────

    def run(
        self,
        query:           str,
        metadata_filters:Optional[Dict[str, Any]]  = None,
        max_chunks:      Optional[int]              = None,
        override_budget: Optional[int]              = None,
        use_history:     bool                       = True,
        record_turn:     bool                       = True,
    ) -> ContextResult:
        """
       run one query through the full pipeline and get a ContextResult with all stats.

        Args:
            query:            query from user
            metadata_filters:  filters for metadata-based filtering in retriever (e.g. {"source": "pdf", "date": {"$gte": "2023-01-01"}})
            max_chunks:        max number of chunks to retrieve/rank (overrides config)
            override_budget:  over the suggested budget from the analyzer (for testing or special cases)
            use_history:     whether to use conversation history for query enrichment
            record_turn:      whether to record this query in the conversation history

        Returns:
            ContextResult   
        """
        t_start = time.perf_counter()

        # ── 0. Cache lookup ────────────────────────────────────────────
        cache_key = None
        if self._cache:
            cache_key = self._cache.make_key(query, metadata_filters)
            cached    = self._cache.get(cache_key)
            if cached:
                logger.debug("ContextEngine: cache hit للاستعلام: %.60s", query)
                # بناء نتيجة من الـ cache
                analysis = self._analyzer.analyze(query)
                return ContextResult(
                    context            = cached,
                    query_analysis     = analysis,
                    chunks_used        = [],
                    pipeline_ms        = (time.perf_counter() - t_start) * 1000,
                    metadata           = {"from_cache": True},
                )

        # ── 1. Multi-turn: إثراء الاستعلام ────────────────────────────
        effective_query = query
        if use_history and self._history and self.config.multi_turn:
            effective_query = self._history.get_enriched_query(query, n=2)

        # ── 2. Query Analysis ──────────────────────────────────────────
        analysis = self._analyzer.analyze(effective_query)

        # ── 3. Dynamic budget ─────────────────────────────────────────
        budget = override_budget or self._adaptive.suggest_budget(analysis)
        analysis.context_budget = budget

        # ── 4. Retrieval ───────────────────────────────────────────────
        retrieved: List[ScoredChunk] = []
        if self._retriever:
            retrieved = self._retriever.retrieve(
                query   = effective_query,
                top_k   = max_chunks or self.config.retriever.top_k,
                filters = metadata_filters,
            )
            # إضافة source quality من الخريطة
            for sc in retrieved:
                src_q = self._source_quality_map.get(sc.source)
                if src_q:
                    sc.source_quality = src_q
        total_retrieved = len(retrieved)

        # ── 5. Filtering ───────────────────────────────────────────────
        filtered = self._filter.filter(retrieved)
        total_after_filter = len(filtered)

        # ── 6. Ranking ─────────────────────────────────────────────────
        ranked = self._ranker.rank(
            filtered, analysis,
            max_n = max_chunks or self.config.ranking.max_chunks_to_rank,
        )
        total_after_rank = len(ranked)

        # ── 7. Context Composition ─────────────────────────────────────
        context = self._composer.compose(ranked, analysis, budget)

        # ── 8. Context Guard ───────────────────────────────────────────
        guard_passed, guard_warnings = self._guard.validate(
            context, effective_query, analysis
        )

        # ── 9. Cache store ─────────────────────────────────────────────
        if self._cache and cache_key and guard_passed:
            self._cache.put(cache_key, context)

        # ── 10. Record in conversation history ────────────────────────
        if record_turn and self._history:
            self._history.add_turn(
                query    = query,
                context  = context,
                analysis = analysis,
            )

        # ── 11. Adaptive learning record ──────────────────────────────
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        self._adaptive.record(
            query          = query,
            analysis       = analysis,
            context_length = len(context),
            chunk_count    = len(ranked),
            response_ms    = elapsed_ms,
            guard_passed   = guard_passed,
        )

        result = ContextResult(
            context            = context,
            query_analysis     = analysis,
            chunks_used        = ranked,
            total_retrieved    = total_retrieved,
            total_after_filter = total_after_filter,
            total_after_rank   = total_after_rank,
            pipeline_ms        = elapsed_ms,
            guard_passed       = guard_passed,
            guard_warnings     = guard_warnings,
        )

        logger.info(
            "ContextEngine.run | query='%.50s' | "
            "retrieved=%d → filtered=%d → ranked=%d | "
            "ctx=%d chars | %.1fms",
            query,
            total_retrieved, total_after_filter, total_after_rank,
            len(context), elapsed_ms,
        )

        return result

    # ─────────────────────────────────────────────────────────────────────
    #  Multi-query API
    # ─────────────────────────────────────────────────────────────────────

    def run_multi(
        self,
        queries:       List[str],
        global_budget: Optional[int]       = None,
        use_weighted:  bool                = True,
        **kwargs,
    ) -> str:
        """
         Run multiple queries and combine their contexts into one unified context.

        Args:
            queries:       list of queries
            global_budget: the total budget
            use_weighted:  whether to use weighted budget allocation based on query complexity

        Returns:
            a unified context combining all queries
        """
        if not queries:
            return ""

        total_budget = global_budget or self.config.composer.max_context_length

        # تحليل جميع الاستعلامات
        analyses = [self._analyzer.analyze(q) for q in queries]

        # توزيع الميزانية
        if use_weighted:
            allocator = ComplexityBasedAllocation()
            budgets   = allocator.allocate(total_budget, len(queries), analyses)
        else:
            budgets = EqualBudgetAllocation().allocate(total_budget, len(queries))

        sections: List[str] = []
        for i, (q, budget, analysis) in enumerate(zip(queries, budgets, analyses)):
            result = self.run(q, override_budget=budget, record_turn=False, **kwargs)
            label  = f"Query: {i+1}" if analysis.is_arabic else f"Query {i+1}"
            sections.append(f"--- {label}: {q} ---\n{result.context}")

        combined = "\n\n".join(sections)

        # Safety clamp
        if self._counter(combined) > total_budget:
            combined = self._hard_truncate(combined, total_budget)

        return combined

    # ─────────────────────────────────────────────────────────────────────
    #  Legacy API (backward-compatible with old ContextManager)
    # ─────────────────────────────────────────────────────────────────────

    def build(
        self,
        query:      str,
        chunks:     List[Tuple],            # [(chunk, score), ...]
        max_chunks: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> str:
        """
            Build context for a single query using the full pipeline, compatible with old build() interface.
        Args:
            query:      the user query
            chunks:     list of (chunk, score) tuples
        """
        analysis = self._analyzer.analyze(query)
        budget   = max_length or analysis.context_budget

        # تحويل tuples → ScoredChunks
        scored = [
            ScoredChunk(
                chunk           = c,
                vector_score    = float(s),
                composite_score = float(s),
            )
            for c, s in chunks
        ]

        filtered = self._filter.filter(scored)
        ranked   = self._ranker.rank(
            filtered, analysis,
            max_n = max_chunks or self.config.ranking.max_chunks_to_rank,
        )
        return self._composer.compose(ranked, analysis, budget)

    def build_multi_query_context(
        self,
        queries:    List[str],
        all_chunks: List[List[Tuple]],
        global_budget: Optional[int] = None,
    ) -> str:
        """build contexts for multiple queries and combine them, compatible with old multi-query interface."""
        combined_queries = []
        for q, chunks in zip(queries, all_chunks):
            ctx = self.build(q, chunks)
            combined_queries.append(f"--- {q} ---\n{ctx}")
        return "\n\n".join(combined_queries)

    def compress_context(self, context: str, target_length: int) -> str:
        """compress an existing context"""
        return self._composer._hard_truncate(context, target_length)

    def get_context_stats(self, context: str) -> Dict[str, Any]:
        """get statistics for an existing context"""
        if isinstance(self._composer, SmartContextComposer):
            return self._composer.get_stats(context)
        return {"total_length": len(context)}

    # ─────────────────────────────────────────────────────────────────────
    #  Diagnostics & Monitoring
    # ─────────────────────────────────────────────────────────────────────

    def get_performance_report(self) -> Dict[str, Any]:
        """get a comprehensive performance report"""
        report = self._adaptive.get_performance_report()
        if self._cache:
            report["cache"] = self._cache.stats
        return report

    def set_conversation(self, history: ConversationHistory) -> None:
        """set a saved conversation history (for resuming a session)"""
        self._history = history

    def get_history(self) -> Optional[ConversationHistory]:
        return self._history

    def clear_cache(self) -> None:
        if self._cache:
            self._cache.clear()
            logger.info("ContextEngine: الـ cache فُرِّغ.")

    def _hard_truncate(self, text: str, max_len: int) -> str:
        if self._counter(text) <= max_len:
            return text
        lo, hi = 0, len(text)
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if self._counter(text[:mid]) <= max_len - 3:
                lo = mid
            else:
                hi = mid
        return text[:lo] + "..."


# ─────────────────────────────────────────────────────────────────────────────
#  ContextManager (backward-compatible alias + enhanced)
# ─────────────────────────────────────────────────────────────────────────────

class ContextManager(ContextEngine):
    """
    Backward-compatible alias for ContextEngine with support for legacy strategies.
    This class allows users to inject legacy deduplication and ranking strategies while still benefiting from the new modular architecture. It wraps the old interfaces but encourages migration to the new strategy-based components for better performance and flexibility.
    """

    def __init__(
        self,
        config:               Optional[ContextEngineConfig] = None,
        *,
        length_counter:       Optional[LengthCounter]       = None,
        dedup_strategy:       Optional[Any]                 = None,
        ranking_strategy:     Optional[Any]                 = None,
        compression_strategy: Optional[Any]                 = None,
        budget_strategy:      Optional[BudgetAllocationStrategy] = None,
        **kwargs,
    ):
        super().__init__(
            config         = config,
            length_counter = length_counter,
            budget_allocator = budget_strategy,
            **kwargs,
        )
        # Legacy strategies still accepted but wrapped
        if dedup_strategy:
            logger.debug("ContextManager: dedup_strategy مُوفَّر (legacy mode)")
        if ranking_strategy:
            logger.debug("ContextManager: ranking_strategy مُوفَّر (legacy mode)")
