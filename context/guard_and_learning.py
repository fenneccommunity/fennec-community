"""
guard_and_learning.py — Context Guard + Adaptive Learning + Budget Allocation
حارس السياق + التعلم التكيّفي + توزيع الميزانية
"""

from __future__ import annotations

import logging
import math
import re
import time
from collections import defaultdict, deque
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

from .config import ContextEngineConfig
from .models import (
     QueryAnalysis, QueryComplexity,
)
from .strategy import BudgetAllocationStrategy, ContextGuardStrategy

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  CONTEXT GUARD
# ─────────────────────────────────────────────────────────────────────────────

class ContextGuard(ContextGuardStrategy):
    """
      check from context validity and relevance before answering or sending to LLM.
      checks include:
      1. empty context
      2. min length
      3. max length
      4. source coverage (presence of [source: ...] or ---)
      5. optional relevance check via embedding similarity (if embed_fn provided)
    """

    def __init__(
        self,
        config:   ContextEngineConfig,
        embed_fn: Optional[Callable[[str], np.ndarray]] = None,
    ):
        self.cfg      = config.guard
        self._embed   = embed_fn
        self._enabled = config.guard.enabled

    def validate(
        self,
        context:  str,
        query:    str,
        analysis: QueryAnalysis,
    ) -> Tuple[bool, List[str]]:
        """
       check for context validity and relevance before answering or sending to LLM.

        Returns:
            (passed: bool, warnings: List[str])
        """
        if not self._enabled:
            return True, []

        warnings: List[str] = []
        passed = True

        # 1. فحص الفراغ
        if self.cfg.check_empty and not context.strip():
            warnings.append("context is empty ⚠")
            return False, warnings

        ctx_len = len(context)

        # 2. فحص الطول الأدنى
        if ctx_len < self.cfg.min_context_length:
            warnings.append(
                f"context is too short ({ctx_len} characters < {self.cfg.min_context_length})."
            )
            passed = False

        # 3. فحص الطول الأقصى
        if ctx_len > self.cfg.max_context_length:
            warnings.append(
                f"context is too long ({ctx_len} > {self.cfg.max_context_length}). "
                "consider compressing it."
            )
            # هذا تحذير فقط — لا يرسب الفحص

        # 4. فحص وجود مصدر
        if self.cfg.check_source_coverage:
            has_source = bool(
                re.search(r"\[(?:|Source|source):", context)
                or re.search(r"---", context)
            )
            if not has_source:
                warnings.append("no valid source found in context.")

        # 5. فحص الصلة الدلالية (اختياري)
        if self.cfg.check_relevance and self._embed is not None:
            rel_ok, rel_msg = self._check_relevance(context, query)
            if not rel_ok:
                warnings.append(rel_msg)
                passed = False

        if warnings:
            logger.warning("ContextGuard: %s", " | ".join(warnings))
        else:
            logger.debug("ContextGuard: ✓ context is valid (length=%d)", ctx_len)

        return passed, warnings

    def _check_relevance(self, context: str, query: str) -> Tuple[bool, str]:
        """فحص صلة السياق بالاستعلام عبر cosine similarity"""
        try:
            q_emb  = self._embed(query)
            c_emb  = self._embed(context[:512])  # نموذج sample للسياق
            q_flat = q_emb.flatten()
            c_flat = c_emb.flatten()
            denom  = np.linalg.norm(q_flat) * np.linalg.norm(c_flat)
            sim    = float(np.dot(q_flat, c_flat) / denom) if denom > 0 else 0.0

            if sim < self.cfg.relevance_threshold:
                return False, (
                    f"context is not relevant to the query (similarity={sim:.3f} < {self.cfg.relevance_threshold})."
                )
            return True, ""
        except Exception as e:
            logger.debug("ContextGuard: failed to check relevance: %s", e)
            return True, ""  # لا نُرسب عند الفشل


# ─────────────────────────────────────────────────────────────────────────────
#  ADAPTIVE LEARNING SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveLearningSystem:
    """
     adaptive learning system that monitors query performance and guard outcomes to suggest context budget adjustments over time.
     what it does:
        1. records query performance metrics (response time, context length, guard failures)
        2. suggests context budget adjustments based on historical performance (e.g. if guard failures are high, suggest reducing budget)
        3. provides periodic performance reports for analysis
     this system helps to optimize context usage dynamically based on real-world feedback.
    """

    WINDOW = 100  # نافذة آخر N استعلام لحساب الإحصائيات

    def __init__(self, config: ContextEngineConfig):
        self.cfg     = config
        self._enabled = config.enable_adaptive

        # إحصائيات rolling
        self._response_times:    Deque[float] = deque(maxlen=self.WINDOW)
        self._context_lengths:   Deque[int]   = deque(maxlen=self.WINDOW)
        self._chunk_counts:      Deque[int]   = deque(maxlen=self.WINDOW)
        self._guard_failures:    Deque[bool]  = deque(maxlen=self.WINDOW)

        # نتائج تعديل الأداء
        self._suggested_budgets: Deque[int]   = deque(maxlen=self.WINDOW)

        # عداد الاستعلامات
        self._total_queries = 0

    def record(
        self,
        query:          str,
        analysis:       QueryAnalysis,
        context_length: int,
        chunk_count:    int,
        response_ms:    float,
        guard_passed:   bool,
    ) -> None:
        """record query performance and guard outcome for adaptive learning"""
        if not self._enabled:
            return

        self._response_times.append(response_ms)
        self._context_lengths.append(context_length)
        self._chunk_counts.append(chunk_count)
        self._guard_failures.append(not guard_passed)
        self._total_queries += 1

        # تعديل تلقائي دوري (كل 20 استعلام)
        if self._total_queries % 20 == 0:
            self._auto_tune()

    def suggest_budget(self, analysis: QueryAnalysis) -> int:
        """
        suggest an optimized context budget based on historical statistics.
        it blends the budget calculated by QueryAnalyzer with the actual usage average.
        """
        base_budget = analysis.context_budget

        if not self._enabled or len(self._context_lengths) < 10:
            return base_budget

        avg_actual = sum(self._context_lengths) / len(self._context_lengths)

        # إذا كانت الميزانية الفعلية المستخدمة أقل بكثير → قلّل
        if avg_actual < base_budget * 0.5:
            adjusted = int(avg_actual * 1.3)  # هامش 30%
            logger.debug(
                "AdaptiveLearning: adjusting budget %d → %d",
                base_budget, adjusted,
            )
            return max(adjusted, 200)

        return base_budget

    def get_performance_report(self) -> Dict[str, Any]:
        """t report on system performance"""
        def _avg(d: deque) -> float:
            return sum(d) / len(d) if d else 0.0

        return {
            "total_queries":      self._total_queries,
            "avg_response_ms":    round(_avg(self._response_times), 2),
            "avg_context_length": round(_avg(self._context_lengths), 1),
            "avg_chunk_count":    round(_avg(self._chunk_counts), 2),
            "guard_failure_rate": round(_avg(self._guard_failures), 3),
            "samples":            len(self._response_times),
        }

    def _auto_tune(self) -> None:
        """automatic adjustment of parameters"""
        avg_rt = sum(self._response_times) / len(self._response_times)
        fail_r = sum(self._guard_failures) / len(self._guard_failures)

        # إذا ارتفع معدل الفشل → سجّل تحذير
        if fail_r > 0.3:
            logger.warning(
                "AdaptiveLearning: Guard failure rate is high (%.1f%%) — "
                "please check the quality of the retrieval system.", fail_r * 100
            )

        # إذا ارتفع زمن الاستجابة → سجّل اقتراح
        if avg_rt > 2000:
            logger.info(
                "AdaptiveLearning: Average response time %.0fms — "
                "consider enabling caching.", avg_rt
            )


# ─────────────────────────────────────────────────────────────────────────────
#  BUDGET ALLOCATION (multi-query)
# ─────────────────────────────────────────────────────────────────────────────

class EqualBudgetAllocation(BudgetAllocationStrategy):
    """equal distribution of context budget across multiple queries (default)"""

    def allocate(self, total_budget: int, n_queries: int) -> List[int]:
        if n_queries <= 0:
            return []
        per = total_budget // n_queries
        budgets = [per] * n_queries
        budgets[0] += total_budget - sum(budgets)
        return budgets


class WeightedBudgetAllocation(BudgetAllocationStrategy):
    """weighted distribution of context budget based on query importance"""

    def __init__(self, weights: List[float]):
        self.weights = weights

    def allocate(self, total_budget: int, n_queries: int) -> List[int]:
        if len(self.weights) != n_queries:
            raise ValueError("weights length must equal n_queries.")
        total_w = sum(self.weights) or 1.0
        raw     = [total_budget * w / total_w for w in self.weights]
        budgets = [max(1, int(b)) for b in raw]
        budgets[0] += total_budget - sum(budgets)
        return budgets


class ComplexityBasedAllocation(BudgetAllocationStrategy):
    """
    Allocate context budget based on the complexity of each query.
    More complex queries automatically receive a larger budget.
    """

    _COMPLEXITY_WEIGHTS = {
        QueryComplexity.LOW:    1.0,
        QueryComplexity.MEDIUM: 1.5,
        QueryComplexity.HIGH:   2.5,
    }

    def allocate(
        self,
        total_budget:    int,
        n_queries:       int,
        analyses:        Optional[List[QueryAnalysis]] = None,
    ) -> List[int]:
        if not analyses or len(analyses) != n_queries:
            return EqualBudgetAllocation().allocate(total_budget, n_queries)

        weights = [
            self._COMPLEXITY_WEIGHTS.get(a.complexity, 1.5)
            for a in analyses
        ]
        return WeightedBudgetAllocation(weights).allocate(total_budget, n_queries)



# ─────────────────────────────────────────────────────────────────────────────
#  RESULT CACHE (LRU)
# ─────────────────────────────────────────────────────────────────────────────

class ContextCache:
    """
    LRU Cache for compound contexts.
    Overrides context reconstruction for repeated queries.
    """

    def __init__(self, max_size: int = 256, ttl_seconds: int = 300):
        self._max     = max_size
        self._ttl     = ttl_seconds
        self._store:  Dict[str, Tuple[str, float]] = {}  # key → (context, timestamp)
        self._order:  deque = deque()
        self._hits    = 0
        self._misses  = 0

    def get(self, key: str) -> Optional[str]:
        if key not in self._store:
            self._misses += 1
            return None
        ctx, ts = self._store[key]
        if time.time() - ts > self._ttl:
            del self._store[key]
            self._misses += 1
            return None
        self._hits += 1
        return ctx

    def put(self, key: str, context: str) -> None:
        if key in self._store:
            self._store[key] = (context, time.time())
            return
        if len(self._store) >= self._max:
            oldest = self._order.popleft()
            self._store.pop(oldest, None)
        self._store[key] = (context, time.time())
        self._order.append(key)

    def make_key(self, query: str, filters: Optional[Dict] = None) -> str:
        import hashlib
        raw = query.strip().lower()
        if filters:
            raw += str(sorted(filters.items()))
        return hashlib.md5(raw.encode()).hexdigest()

    @property
    def stats(self) -> Dict[str, Any]:
        total = self._hits + self._misses
        return {
            "size":      len(self._store),
            "max_size":  self._max,
            "hits":      self._hits,
            "misses":    self._misses,
            "hit_rate":  f"{self._hits/total:.1%}" if total else "0%",
            "ttl_sec":   self._ttl,
        }

    def clear(self) -> None:
        self._store.clear()
        self._order.clear()
