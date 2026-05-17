"""
 Filtering + Ranking + Context Composer
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import ContextEngineConfig, LengthCounter
from .models import (
    QueryAnalysis, ScoredChunk
)
from .strategy import  ChunkRankingStrategy, ContextComposerStrategy

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Utility
# ─────────────────────────────────────────────────────────────────────────────

def char_counter(text: str) -> int:
    return len(text) if text else 0


def make_token_counter(tokenizer) -> LengthCounter:
    def _count(text: str) -> int:
        return len(tokenizer.encode(text))
    return _count


# ─────────────────────────────────────────────────────────────────────────────
#  FILTERING LAYER
# ─────────────────────────────────────────────────────────────────────────────

class ChunkFilter:
    """
   chunks filtering layer.
   tasks:
   1.remove noise (regex patterns)
   2.filter by length (min/max)
   3.dedup by content hash
   4.dedup semantically (optional, with embeddings)
    """

    def __init__(self, config: ContextEngineConfig):
        self.cfg         = config.filter_cfg
        self._noise_res  = [re.compile(p) for p in self.cfg.noise_patterns]

    # ── Public API ────────────────────────────────────────────────────────

    def filter(self, chunks: List[ScoredChunk]) -> List[ScoredChunk]:
        before = len(chunks)
        chunks = self._filter_noise(chunks)
        chunks = self._filter_length(chunks)
        chunks = self._dedup_hash(chunks)
        after  = len(chunks)
        if before != after:
            logger.debug("ChunkFilter: %d → %d (حُذف %d)", before, after, before - after)
        return chunks

    def filter_with_embeddings(
        self,
        chunks:     List[ScoredChunk],
        embeddings: List[Optional[np.ndarray]],
    ) -> List[ScoredChunk]:
        """filter + semantic deduplication"""
        chunks = self.filter(chunks)
        chunks = self._dedup_semantic(chunks, embeddings)
        return chunks

    # ── Private helpers ───────────────────────────────────────────────────

    def _filter_noise(self, chunks: List[ScoredChunk]) -> List[ScoredChunk]:
        clean = []
        for sc in chunks:
            is_noise = any(p.match(sc.text.strip()) for p in self._noise_res)
            if not is_noise:
                clean.append(sc)
        return clean

    def _filter_length(self, chunks: List[ScoredChunk]) -> List[ScoredChunk]:
        return [
            sc for sc in chunks
            if self.cfg.min_chunk_length <= sc.char_count <= self.cfg.max_chunk_length
        ]

    def _dedup_hash(self, chunks: List[ScoredChunk]) -> List[ScoredChunk]:
        seen: set = set()
        unique  = []
        for sc in chunks:
            if sc.content_hash not in seen:
                seen.add(sc.content_hash)
                unique.append(sc)
        return unique

    def _dedup_semantic(
        self,
        chunks:     List[ScoredChunk],
        embeddings: List[Optional[np.ndarray]],
    ) -> List[ScoredChunk]:
        """remove semantically similar chunks based on cosine similarity of embeddings"""
        if len(chunks) != len(embeddings):
            return chunks

        threshold = self.cfg.semantic_sim_threshold
        keep = [True] * len(chunks)
        valid = [(i, e) for i, e in enumerate(embeddings) if e is not None]

        for i in range(len(valid)):
            if not keep[valid[i][0]]:
                continue
            ei = valid[i][1].flatten()
            for j in range(i + 1, len(valid)):
                if not keep[valid[j][0]]:
                    continue
                ej = valid[j][1].flatten()
                denom = np.linalg.norm(ei) * np.linalg.norm(ej)
                if denom == 0:
                    continue
                if float(np.dot(ei, ej) / denom) >= threshold:
                    keep[valid[j][0]] = False

        return [sc for sc, k in zip(chunks, keep) if k]


# ─────────────────────────────────────────────────────────────────────────────
#  RANKING SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

class CompositeRanker(ChunkRankingStrategy):
    """
    composite ranker system.
    calculates a composite score for each chunk based on multiple factors:
      - semantic similarity (vector_score)
      - keyword matching score
      - source quality
      - recency
      - position (retrieval rank in the original doc - earlier is better )
    then sorts chunks by this composite score and returns the top N.
    """

    def __init__(self, config: ContextEngineConfig):
        self.cfg = config.ranking

    def rank(
        self,
        chunks:   List[ScoredChunk],
        analysis: QueryAnalysis,
        max_n:    Optional[int] = None,
    ) -> List[ScoredChunk]:
        if not chunks:
            return []

        for sc in chunks:
            sc.composite_score = self._compute_composite(sc, analysis)

        ranked = sorted(chunks, key=lambda x: x.composite_score, reverse=True)
        limit  = max_n or self.cfg.max_chunks_to_rank
        result = ranked[:limit]

        logger.debug(
            "CompositeRanker: %d → %d | top score=%.4f",
            len(chunks), len(result),
            result[0].composite_score if result else 0,
        )
        return result

    def _compute_composite(
        self, sc: ScoredChunk, analysis: QueryAnalysis
    ) -> float:
        w = self.cfg

        # Source quality (normalize SourceQuality enum value to [0,1])
        sq_val = sc.source_quality.value / 3.0  # 0→0, 1→0.33, 2→0.67, 3→1.0

        # Keyword boost: count query keywords found in chunk text
        kw_boost = self._keyword_boost(sc.text, analysis.keywords)

        # Position score: chunks earlier in doc slightly preferred
        pos_score = max(0.0, 1.0 - sc.retrieval_rank * 0.01)

        composite = (
            sc.vector_score   * w.weight_vector_score   +
            kw_boost          * w.weight_keyword_score  +
            sq_val            * w.weight_source_quality +
            sc.recency_score  * w.weight_recency        +
            pos_score         * w.weight_position
        )
        return round(composite, 6)

    @staticmethod
    def _keyword_boost(text: str, keywords: List[str]) -> float:
        if not keywords or not text:
            return 0.0
        text_l  = text.lower()
        matches = sum(1 for kw in keywords if kw in text_l)
        return min(matches / len(keywords), 1.0)


# ─────────────────────────────────────────────────────────────────────────────
#  CONTEXT COMPOSER
# ─────────────────────────────────────────────────────────────────────────────

class ContextTemplate:
    """templates for formatting the composed context in different styles (arabic, english, minimal, structured)"""
    def __init__(
        self,
        header:           str,
        chunk_fmt:        str,
        chunk_fmt_score:  str,
        footer:           str,
    ):
        self.header          = header
        self.chunk_fmt       = chunk_fmt
        self.chunk_fmt_score = chunk_fmt_score
        self.footer          = footer


_TEMPLATES: Dict[str, ContextTemplate] = {
    "arabic": ContextTemplate(
        header          = "📚 المعلومات المسترجعة:\n",
        chunk_fmt       = "[المصدر: {source}]\n{text}",
        chunk_fmt_score = "[المصدر: {source} | الدرجة: {score:.3f}]\n{text}",
        footer          = "\n─── نهاية السياق ───",
    ),
    "english": ContextTemplate(
        header          = "📚 Retrieved Information:\n",
        chunk_fmt       = "[Source: {source}]\n{text}",
        chunk_fmt_score = "[Source: {source} | Score: {score:.3f}]\n{text}",
        footer          = "\n─── End of Context ───",
    ),
    "minimal": ContextTemplate(
        header          = "",
        chunk_fmt       = "{text}",
        chunk_fmt_score = "{text}",
        footer          = "",
    ),
    "structured": ContextTemplate(
        header          = "=== CONTEXT START ===\n",
        chunk_fmt       = "--- [{idx}] {source} ---\n{text}",
        chunk_fmt_score = "--- [{idx}] {source} (score={score:.3f}) ---\n{text}",
        footer          = "\n=== CONTEXT END ===",
    ),
}


class SmartContextComposer(ContextComposerStrategy):
    """
   smart context composer that formats and assembles the ranked chunks into a final context string.

   tasks:
   1. format each chunk with source and optional score using a template
   2. assemble chunks with separators, ensuring total length does not exceed budget
   3. support different templates/styles (arabic, english, minimal, structured)
   4. optional grouping of chunks by source
   5. provide stats about the composed context (total length, chunk count, etc.)

    """

    def __init__(
        self,
        config:  ContextEngineConfig,
        counter: Optional[LengthCounter] = None,
    ):
        self.cfg     = config.composer
        self._counter = counter or char_counter
        self._tmpl   = _TEMPLATES.get(self.cfg.template, _TEMPLATES["arabic"])

    # ── Public API ────────────────────────────────────────────────────────

    def compose(
        self,
        chunks:   List[ScoredChunk],
        analysis: QueryAnalysis,
        budget:   int,
    ) -> str:
        if not chunks:
            return self._empty_context()

        # اختياري: تجميع حسب المصدر
        if self.cfg.group_by_source:
            chunks = self._group_by_source(chunks)

        # تنسيق كل chunk
        formatted: List[Tuple[str, ScoredChunk]] = []
        for idx, sc in enumerate(chunks):
            text = self._format_chunk(sc, idx)
            formatted.append((text, sc))

        # تجميع greedy مع ضمان عدم تجاوز budget
        context = self._assemble(formatted, budget)

        logger.debug(
            "SmartContextComposer: %d chunks → %d chars (budget=%d)",
            len(chunks), self._counter(context), budget,
        )
        return context

    # ── Private helpers ───────────────────────────────────────────────────

    def _format_chunk(self, sc: ScoredChunk, idx: int) -> str:
        tmpl = self._tmpl
        fmt  = tmpl.chunk_fmt_score if self.cfg.include_scores else tmpl.chunk_fmt

        # دعم {idx} في structured template
        text = fmt.format(
            source = sc.source or sc.doc_id or f"doc-{idx}",
            text   = sc.text.strip(),
            score  = sc.composite_score,
            idx    = idx + 1,
        )

        if self.cfg.include_metadata and sc.metadata:
            meta_str = " | ".join(
                f"{k}: {v}" for k, v in list(sc.metadata.items())[:5]
            )
            text = f"{text}\n[{meta_str}]"

        return text

    def _group_by_source(self, chunks: List[ScoredChunk]) -> List[ScoredChunk]:
        """تجميع الـ chunks من نفس المصدر معاً"""
        groups: Dict[str, List[ScoredChunk]] = defaultdict(list)
        for sc in chunks:
            groups[sc.source].append(sc)
        # رتّب المجموعات بأعلى درجة في المجموعة
        sorted_groups = sorted(
            groups.values(),
            key=lambda g: max(sc.composite_score for sc in g),
            reverse=True,
        )
        return [sc for group in sorted_groups for sc in group]

    def _assemble(
        self,
        formatted: List[Tuple[str, ScoredChunk]],
        budget:    int,
    ) -> str:
        tmpl = self._tmpl
        sep  = self.cfg.separator
        parts: List[str] = []

        if tmpl.header:
            parts.append(tmpl.header.rstrip())

        footer_cost = self._counter(tmpl.footer) + self._counter(sep) if tmpl.footer else 0
        available   = budget - self._counter(sep.join(
            ([tmpl.header.rstrip()] if tmpl.header else [])
        )) - footer_cost

        included = 0
        for chunk_text, _ in formatted:
            chunk_len = self._counter(chunk_text) + self._counter(sep)
            if available - chunk_len < 0:
                if included == 0:
                    # يجب إدراج chunk واحد على الأقل — اقطعها
                    parts.append(self._hard_truncate(chunk_text, available))
                    included += 1
                break
            parts.append(chunk_text)
            available -= chunk_len
            included  += 1

        if tmpl.footer:
            parts.append(tmpl.footer.lstrip())

        context = sep.join(parts)

        # Safety clamp
        if self._counter(context) > budget:
            context = self._hard_truncate(context, budget)

        return context

    def _empty_context(self) -> str:
        tmpl = self._tmpl
        msg  = "لا تتوفر معلومات." if self.cfg.template == "arabic" else "No information available."
        return self.cfg.separator.join(filter(None, [tmpl.header.strip(), msg, tmpl.footer.strip()]))

    def _hard_truncate(self, text: str, max_len: int) -> str:
        """قطع النص بطريقة ذكية (عند آخر مسافة)"""
        if self._counter(text) <= max_len:
            return text
        # Binary search
        lo, hi = 0, len(text)
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if self._counter(text[:mid]) <= max_len - 3:
                lo = mid
            else:
                hi = mid
        cut   = text[:lo]
        space = cut.rfind(" ")
        if space > lo * 0.7:
            cut = cut[:space]
        return cut + "..."

    def get_stats(self, context: str) -> Dict[str, Any]:
        """stats about the composed context (total length, chunk count, avg/min/max chunk length, etc.)"""
        parts = context.split(self.cfg.separator)
        parts = [p for p in parts if p.strip()]
        lengths = [self._counter(p) for p in parts]
        return {
            "total_length":     self._counter(context),
            "chunk_count":      len(parts),
            "avg_chunk_length": sum(lengths) // len(lengths) if lengths else 0,
            "min_chunk_length": min(lengths) if lengths else 0,
            "max_chunk_length": max(lengths) if lengths else 0,
            "template":         self.cfg.template,
            "length_unit":      "tokens" if self._counter is not char_counter else "chars",
        }
