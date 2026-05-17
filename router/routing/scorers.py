"""

The hybrid scoring layer that combines three signals:

  1. **SemanticScorer**  — cosine similarity on sentence embeddings
  2. **KeywordScorer**   — lightweight keyword/pattern matching
  3. **LLMScorer**       — optional LLM-based classification (costly; used as fallback)

All scorers implement :class:`BaseScorer` so they can be swapped out
or stacked without touching the pipeline.
"""
from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.route import Route
from ..config import ScoringConfig, SimilarityMetric


# ---------------------------------------------------------------------------
# Base Scorer
# ---------------------------------------------------------------------------

class BaseScorer(ABC):
    """Contract for every scorer.  Returns a float in [0, 1]."""

    @abstractmethod
    def score(self, query: str, route: Route, **ctx) -> float:
        """Return a relevance score in [0.0, 1.0]."""
        ...

    @abstractmethod
    def batch_score(
        self,
        query: str,
        routes: List[Route],
        **ctx,
    ) -> List[float]:
        """Score a query against many routes at once (may be more efficient)."""
        ...

    def name(self) -> str:
        return self.__class__.__name__


# ---------------------------------------------------------------------------
# 1. Semantic Scorer
# ---------------------------------------------------------------------------

class SemanticScorer(BaseScorer):
    """
    Scores via cosine similarity between a query embedding and each route's
    per-example embeddings.

    The final score is the **maximum** similarity across all examples —
    a query only needs to match *one* example well to trigger the route.
    """

    def __init__(self, cfg: ScoringConfig):
        self._cfg = cfg

    # ---- Similarity primitives ------------------------------------------

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    @staticmethod
    def _dot_product(a: np.ndarray, b: np.ndarray) -> float:
        # Assumes both vectors are already L2-normalised
        return float(np.dot(a, b))

    @staticmethod
    def _euclidean_to_similarity(a: np.ndarray, b: np.ndarray) -> float:
        dist = float(np.linalg.norm(a - b))
        return 1.0 / (1.0 + dist)

    def _sim(self, a: np.ndarray, b: np.ndarray) -> float:
        metric = self._cfg.similarity_metric
        if metric == SimilarityMetric.COSINE:
            return self._cosine(a, b)
        elif metric == SimilarityMetric.DOT_PRODUCT:
            return self._dot_product(a, b)
        else:
            return self._euclidean_to_similarity(a, b)

    # ---- Public interface -----------------------------------------------

    def score(
        self,
        query: str,
        route: Route,
        query_embedding: Optional[List[float]] = None,
        **_,
    ) -> float:
        if not route.embeddings or query_embedding is None:
            return 0.0

        q = np.array(query_embedding, dtype=np.float32)
        max_sim = 0.0
        for emb in route.embeddings:
            r = np.array(emb, dtype=np.float32)
            sim = self._sim(q, r)
            if sim > max_sim:
                max_sim = sim
        return max_sim

    def batch_score(
        self,
        query: str,
        routes: List[Route],
        query_embedding: Optional[List[float]] = None,
        **_,
    ) -> List[float]:
        if query_embedding is None:
            return [0.0] * len(routes)

        q = np.array(query_embedding, dtype=np.float32)
        scores = []
        for route in routes:
            if not route.embeddings:
                scores.append(0.0)
                continue
            # Vectorised: stack all example embeddings, compute all similarities at once
            emb_matrix = np.array(route.embeddings, dtype=np.float32)   # (n_examples, dim)
            if self._cfg.similarity_metric == SimilarityMetric.COSINE:
                norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1e-9, norms)
                emb_norm = emb_matrix / norms
                q_norm   = q / (np.linalg.norm(q) + 1e-9)
                sims     = emb_norm @ q_norm
            elif self._cfg.similarity_metric == SimilarityMetric.DOT_PRODUCT:
                sims = emb_matrix @ q
            else:
                diffs = emb_matrix - q
                dists = np.linalg.norm(diffs, axis=1)
                sims  = 1.0 / (1.0 + dists)

            scores.append(float(np.max(sims)))
        return scores


# ---------------------------------------------------------------------------
# 2. Keyword Scorer
# ---------------------------------------------------------------------------

class KeywordScorer(BaseScorer):
    """
    Lightweight keyword/pattern scorer using the ``RouteKeywords`` config
    attached to each route.

    Scoring rules
    -------------
    - If a *required* keyword is missing  → score = 0.0 (veto)
    - If an *excluded* keyword is present → score = 0.0 (veto)
    - For each *any_of* match: add ``keywords.boost``
    - Result is clamped to [0, 1]

    This runs in O(n_keywords) with no model inference, so it's essentially free.
    """

    def score(self, query: str, route: Route, **_) -> float:
        query_lower = query.lower()
        kw = route.keywords

        # Veto checks
        for word in kw.required:
            if word.lower() not in query_lower:
                return 0.0
        for word in kw.excluded:
            if word.lower() in query_lower:
                return 0.0

        # Accumulate boost
        bonus = 0.0
        for word in kw.any_of:
            if word.lower() in query_lower:
                bonus += kw.boost

        # Normalise: if any_of is empty, return a small base score (route is
        # not vetoed, but keywords didn't add signal)
        if not kw.any_of:
            return 0.0 if (kw.required or kw.excluded) else 0.1

        max_possible = len(kw.any_of) * kw.boost
        return min(1.0, bonus / max_possible if max_possible > 0 else 0.0)

    def batch_score(self, query: str, routes: List[Route], **_) -> List[float]:
        return [self.score(query, r) for r in routes]


# ---------------------------------------------------------------------------
# 3. LLM Scorer  (optional, costly — used only for low-confidence fallback)
# ---------------------------------------------------------------------------

class LLMScorer(BaseScorer):
    """
    Uses a language model to classify the query against all candidate route
    descriptions in a single prompt.

    Invoked **only** when the combined semantic+keyword score falls below
    the ``fallback_confidence_threshold``.

    The LLM returns a JSON mapping {route_name: score (0–1)}.
    """

    def __init__(self, model: str, api_key: Optional[str] = None):
        self._model   = model
        self._api_key = api_key
        self._client  = self._build_client()

    def _build_client(self):
        """Build the LLM client lazily so missing deps don't break import."""
        try:
            import openai
            client = openai.OpenAI(api_key=self._api_key) if self._api_key else openai.OpenAI()
            return client
        except ImportError:
            return None

    def _classify(self, query: str, routes: List[Route]) -> Dict[str, float]:
        if self._client is None:
            return {}

        descriptions = "\n".join(
            f'- "{r.name}": {r.description}' for r in routes
        )
        prompt = (
            f"Given the user query below, rate how well it matches each route "
            f"on a scale from 0.0 (no match) to 1.0 (perfect match).\n\n"
            f"Query: {query}\n\n"
            f"Routes:\n{descriptions}\n\n"
            f"Respond ONLY with a JSON object like: "
            f'{{"{routes[0].name}": 0.9, ...}}'
        )
        try:
            resp = self._client.chat.completions.create(
                model    = self._model,
                messages = [{"role": "user", "content": prompt}],
                max_tokens = 200,
                response_format = {"type": "json_object"},
            )
            import json
            return json.loads(resp.choices[0].message.content or "{}")
        except Exception:
            return {}

    def score(self, query: str, route: Route, **ctx) -> float:
        routes = ctx.get("all_routes", [route])
        scores = self._classify(query, routes)
        return float(scores.get(route.name, 0.0))

    def batch_score(self, query: str, routes: List[Route], **_) -> List[float]:
        scores = self._classify(query, routes)
        return [float(scores.get(r.name, 0.0)) for r in routes]


# ---------------------------------------------------------------------------
# Hybrid Scorer  (the compositor)
# ---------------------------------------------------------------------------

class HybridScorer:
    """
    Combines semantic, keyword, and optional LLM scores using configurable weights.

    Score formula::

        combined = (w_sem * semantic + w_kw * keyword + w_llm * llm) + route.score_bias

    An LLM scorer is only invoked when the semantic+keyword combined falls
    below ``cfg.fallback_confidence_threshold`` to save cost/latency.
    """

    def __init__(
        self,
        cfg:         ScoringConfig,
        semantic:    SemanticScorer,
        keyword:     KeywordScorer,
        llm:         Optional[LLMScorer] = None,
    ):
        self._cfg      = cfg
        self._semantic = semantic
        self._keyword  = keyword
        self._llm      = llm

        # Normalise weights so they sum to 1
        llm_w = cfg.llm_weight if llm else 0.0
        total = cfg.semantic_weight + cfg.keyword_weight + llm_w
        self._w_sem = cfg.semantic_weight / total
        self._w_kw  = cfg.keyword_weight  / total
        self._w_llm = llm_w / total

    def score_all(
        self,
        query:           str,
        routes:          List[Route],
        query_embedding: Optional[List[float]] = None,
    ) -> List[Tuple[Route, float, float, float]]:
        """
        Score all routes and return list of (route, semantic, keyword, combined).
        The list is sorted by effective (combined + bias) score descending.
        """
        sem_scores = self._semantic.batch_score(
            query, routes, query_embedding=query_embedding
        )
        kw_scores  = self._keyword.batch_score(query, routes)

        results: List[Tuple[Route, float, float, float]] = []

        # Check if any route is below fallback threshold before LLM call
        raw_combined = [
            self._w_sem * s + self._w_kw * k
            for s, k in zip(sem_scores, kw_scores)
        ]
        best_raw = max(raw_combined) if raw_combined else 0.0

        llm_scores = [0.0] * len(routes)
        if (
            self._llm is not None
            and best_raw < self._cfg.fallback_confidence_threshold
        ):
            llm_scores = self._llm.batch_score(
                query, routes, all_routes=routes
            )

        for route, sem, kw, llm, raw in zip(
            routes, sem_scores, kw_scores, llm_scores, raw_combined
        ):
            combined = self._w_sem * sem + self._w_kw * kw + self._w_llm * llm
            effective = min(1.0, combined + route.score_bias)
            results.append((route, sem, kw, effective))

        results.sort(key=lambda x: x[3], reverse=True)
        return results
