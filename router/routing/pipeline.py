"""

The ``RoutingPipeline`` orchestrates the full routing decision process:

  1. Encode query → embedding
  2. Select candidate routes from the active RouteGroups
  3. Score candidates with HybridScorer
  4. Evaluate confidence and determine the action (route / fallback / clarify)
  5. Return a ranked list of RouteCandidate objects + a RoutingTrace

The pipeline is *stateless* with respect to routes — it receives a snapshot
of active routes on each call.  State (feedback scores, cache) lives in
other components injected via the constructor.
"""
from __future__ import annotations

import hashlib
import time
from typing import Dict, List, Optional, Tuple

from ..config import ConfidenceLevel, RouterConfig, ScoringConfig
from ..core.result import RouteCandidate, RoutingTrace
from ..core.route import Route
from ..core.route_group import RouteGroup
from .scorers import HybridScorer


# ---------------------------------------------------------------------------
# Confidence Evaluator
# ---------------------------------------------------------------------------

class ConfidenceEvaluator:
    """
    Translates a raw numeric score into a :class:`ConfidenceLevel` and
    decides what action the pipeline should take.
    """

    def __init__(self, cfg: ScoringConfig):
        self._high   = cfg.high_confidence_threshold
        self._medium = cfg.medium_confidence_threshold
        self._fallback = cfg.fallback_confidence_threshold

    def evaluate(self, score: float) -> ConfidenceLevel:
        if score >= self._high:
            return ConfidenceLevel.HIGH
        if score >= self._medium:
            return ConfidenceLevel.MEDIUM
        if score >= self._fallback:
            return ConfidenceLevel.LOW
        return ConfidenceLevel.NONE

    def should_use_llm(self, score: float) -> bool:
        return score < self._fallback

    def should_route(self, level: ConfidenceLevel) -> bool:
        return level in (ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM, ConfidenceLevel.LOW)


# ---------------------------------------------------------------------------
# Embedding Provider  (thin wrapper over SentenceTransformer)
# ---------------------------------------------------------------------------

class EmbeddingProvider:
    """
    Wraps a sentence-transformer model with an in-memory embedding cache.
    Batch encoding is used whenever multiple texts need encoding at once.
    """

    def __init__(self, model_name: str, batch_size: int = 32):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            ) from exc

        self._model      = SentenceTransformer(model_name)
        self._batch_size = batch_size
        self._cache: Dict[str, List[float]] = {}   # query → embedding cache

    def encode(self, text: str, use_cache: bool = True) -> List[float]:
        key = hashlib.md5(text.encode()).hexdigest()
        if use_cache and key in self._cache:
            return self._cache[key]

        import numpy as np
        emb = self._model.encode(
            text, convert_to_numpy=True, normalize_embeddings=True
        ).tolist()

        if use_cache:
            self._cache[key] = emb
        return emb

    def encode_batch(self, texts: List[str], use_cache: bool = True) -> List[List[float]]:
        """Encode many texts, using the cache where possible."""
        import numpy as np

        keys     = [hashlib.md5(t.encode()).hexdigest() for t in texts]
        results  = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts:   List[str] = []

        for i, (text, key) in enumerate(zip(texts, keys)):
            if use_cache and key in self._cache:
                results[i] = self._cache[key]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            new_embs = self._model.encode(
                uncached_texts,
                convert_to_numpy   = True,
                normalize_embeddings = True,
                batch_size         = self._batch_size,
            ).tolist()

            for idx, emb, key in zip(uncached_indices, new_embs, [keys[i] for i in uncached_indices]):
                results[idx] = emb
                if use_cache:
                    self._cache[key] = emb

        return results  # type: ignore[return-value]

    def encode_route(self, route: Route):
        """Compute and store embeddings for all examples in a route."""
        if route.examples:
            route.embeddings = self.encode_batch(route.examples)

    def encode_group(self, group: RouteGroup):
        """Compute group-level intent embeddings."""
        if group.intent_examples:
            group.embeddings = self.encode_batch(group.intent_examples)

    def clear_cache(self):
        self._cache.clear()

    @property
    def cache_size(self) -> int:
        return len(self._cache)


# ---------------------------------------------------------------------------
# Routing Pipeline
# ---------------------------------------------------------------------------

class RoutingPipeline:
    """
    Two-level routing pipeline:

    Level 1 — Group selection
    -------------------------
    Score the query against each RouteGroup's ``intent_examples`` and
    ``intent_keywords``.  Select the top group(s).

    Level 2 — Route selection within groups
    ----------------------------------------
    Score the query against all routes within the selected group(s).
    Surface the top-k candidates with confidence levels.

    The pipeline is designed to be called on every request (the cache layer
    is applied *above* in HierarchicalRouter).
    """

    def __init__(
        self,
        cfg:         RouterConfig,
        embedder:    EmbeddingProvider,
        scorer:      HybridScorer,
        confidence:  ConfidenceEvaluator,
    ):
        self._cfg        = cfg
        self._embedder   = embedder
        self._scorer     = scorer
        self._confidence = confidence

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(
        self,
        query:           str,
        groups:          List[RouteGroup],
        request_id:      str,
        embedding_cache: Optional[Dict[str, List[float]]] = None,
    ) -> Tuple[List[RouteCandidate], RoutingTrace]:
        """
        Execute the full routing pipeline for a query.

        Returns
        -------
        candidates : Ranked list of RouteCandidate (best first).
        trace      : Full audit trail of the routing decision.
        """
        trace = RoutingTrace(request_id=request_id, query=query)

        active_groups = [g for g in groups if g.enabled]
        if not active_groups:
            trace.add_note("No active route groups registered.")
            return [], trace

        # ---- Step 1: Embed query ----------------------------------------
        t0 = time.perf_counter()
        q_emb = self._embedder.encode(query)
        trace.embedding_time_ms = (time.perf_counter() - t0) * 1000

        # ---- Step 2: Select best group(s) --------------------------------
        t1 = time.perf_counter()
        selected_groups = self._select_groups(query, q_emb, active_groups, trace)
        if not selected_groups:
            trace.scoring_time_ms = (time.perf_counter() - t1) * 1000
            trace.add_note("No group matched the query.")
            return [], trace

        # ---- Step 3: Collect & score candidate routes --------------------
        candidate_routes: List[Route] = []
        for group in selected_groups:
            candidate_routes.extend(group.get_routes(enabled_only=True))

        if not candidate_routes:
            trace.scoring_time_ms = (time.perf_counter() - t1) * 1000
            trace.add_note("Selected group(s) have no enabled routes.")
            return [], trace

        scored = self._scorer.score_all(query, candidate_routes, q_emb)
        trace.scoring_time_ms = (time.perf_counter() - t1) * 1000

        # ---- Step 4: Build candidates & apply confidence -----------------
        top_k      = self._cfg.scoring.top_k
        candidates = []

        for route, sem_score, kw_score, combined in scored[:top_k]:
            conf = self._confidence.evaluate(combined)
            candidates.append(RouteCandidate(
                route_name     = route.name,
                group_name     = route.group,
                combined_score = combined,
                semantic_score = sem_score,
                keyword_score  = kw_score,
                llm_score      = 0.0,      # already folded into combined
                confidence     = conf,
                score_bias     = route.score_bias,
            ))

        # Populate trace
        trace.candidates = candidates
        if candidates:
            best = candidates[0]
            trace.selected_route = best.route_name
            trace.selected_group = best.group_name
            trace.confidence     = best.confidence
            if not self._confidence.should_route(best.confidence):
                trace.add_note(
                    f"Best score {best.effective_score:.3f} is below "
                    f"fallback threshold {self._cfg.scoring.fallback_confidence_threshold}."
                )

        return candidates, trace

    # ------------------------------------------------------------------ #
    # Group selection
    # ------------------------------------------------------------------ #

    def _select_groups(
        self,
        query:         str,
        query_emb:     List[float],
        groups:        List[RouteGroup],
        trace:         RoutingTrace,
    ) -> List[RouteGroup]:
        """
        Score each group by its intent examples + keyword signals and
        return the top-scoring group(s).

        If only one group exists, return it directly (avoids double-encoding).
        """
        if len(groups) == 1:
            return groups

        # Build a synthetic Route per group to reuse the scorer
        from ..core.route import Route, RouteKeywords
        from ..core.base import CallableHandler

        proxies: List[Route] = []
        for g in groups:
            proxy = Route(
                name        = g.name,
                description = g.description,
                handler     = CallableHandler(lambda q: None),
                examples    = g.intent_examples,
                keywords    = g.intent_keywords,
                priority    = g.priority,
            )
            proxy.embeddings = g.embeddings   # re-use pre-computed group embeddings
            proxies.append(proxy)

        scored = self._scorer.score_all(query, proxies, query_emb)

        # Pick top group(s) — currently top-1; extend for multi-group queries
        best_proxy, _, _, best_score = scored[0]
        conf = self._confidence.evaluate(best_score)

        # Map proxy name back to group
        group_map = {g.name: g for g in groups}
        selected = [group_map[best_proxy.name]] if best_proxy.name in group_map else []

        if selected:
            trace.add_note(
                f"Group selected: '{selected[0].name}' "
                f"(score={best_score:.3f}, confidence={conf.value})"
            )
        return selected
