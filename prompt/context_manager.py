"""
Context Manager — Intelligent Context Engineering
==================================================
Transforms a raw list of retrieved documents into an optimally-ordered,
de-duplicated, token-budget-aware context block ready for injection into
the prompt.

Key behaviours
--------------
* Relevance-first ordering  — higher-score docs appear closer to the question
* Lost-in-the-middle fix    — most relevant docs placed at start AND end
* Deduplication             — near-identical passages removed (MinHash)
* Adaptive summarization    — docs that don't fit get one-sentence synopses
* Token budget enforcement  — hard cap via simple whitespace tokenizer (swap
                               for tiktoken in production)
* Citation index            — each included doc gets a [1], [2] label
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .types import Document, Message, PromptRequest

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Tokenizer shim  (replace with tiktoken)
# ─────────────────────────────────────────────

def _count_tokens(text: str) -> int:
    """Approximate token count (words × 1.3 is a common heuristic)."""
    return int(len(text.split()) * 1.3)


# ─────────────────────────────────────────────
# Deduplication helpers
# ─────────────────────────────────────────────

def _shingles(text: str, k: int = 5) -> set:
    words = text.lower().split()
    return {" ".join(words[i: i + k]) for i in range(max(len(words) - k + 1, 1))}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _content_hash(text: str) -> str:
    return hashlib.md5(re.sub(r"\s+", " ", text.lower()).encode()).hexdigest()


# ─────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────

@dataclass
class ContextResult:
    context_block:      str                   # formatted, ready-to-inject
    included_docs:      List[Document]
    excluded_docs:      List[Document]
    citation_map:       Dict[int, str]        # index → source
    tokens_used:        int
    tokens_budget:      int
    duplicates_removed: int
    truncated:          bool

    @property
    def utilization_pct(self) -> float:
        return round(self.tokens_used / max(self.tokens_budget, 1) * 100, 1)


# ─────────────────────────────────────────────
# ContextManager
# ─────────────────────────────────────────────

class ContextManager:
    """
    Builds an optimised context block from a list of retrieved documents.

    Usage
    -----
    cm = ContextManager()
    result = cm.build(request)
    # inject result.context_block into the prompt template
    """

    def __init__(
        self,
        dedup_threshold:     float = 0.85,   # Jaccard similarity above which a doc is a duplicate
        min_doc_tokens:      int   = 5,      # skip docs shorter than this
        use_lost_in_middle:  bool  = True,   # reorder to fight attention decay
        summarize_overflow:  bool  = False,  # summarise docs that overflow budget (stub)
        max_memory_messages: int   = 10,     # how many history turns to include
    ) -> None:
        self.dedup_threshold     = dedup_threshold
        self.min_doc_tokens      = min_doc_tokens
        self.use_lost_in_middle  = use_lost_in_middle
        self.summarize_overflow  = summarize_overflow
        self.max_memory_messages = max_memory_messages

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────

    def build(self, request: PromptRequest) -> ContextResult:
        """
        Process ``request.documents`` and return a ``ContextResult``.
        """
        budget  = request.max_context_tokens
        docs    = request.documents

        # 1. Filter too-short docs
        docs = [d for d in docs if _count_tokens(d.content) >= self.min_doc_tokens]

        # 2. Sort by score (descending)
        docs = sorted(docs, key=lambda d: d.score, reverse=True)

        # 3. Deduplicate
        docs, n_dupes = self._deduplicate(docs)

        # 4. "Lost-in-the-middle" reordering
        if self.use_lost_in_middle and len(docs) > 3:
            docs = self._lost_in_middle_reorder(docs)

        # 5. Token budget enforcement
        included, excluded, truncated = self._apply_budget(docs, budget)

        # 6. Format context block + build citation map
        context_block, citation_map = self._format(included)

        tokens_used = _count_tokens(context_block)

        logger.debug(
            "[ContextManager] %d/%d docs included, %d dupes removed, %d tokens used / %d budget",
            len(included), len(docs) + n_dupes, n_dupes, tokens_used, budget,
        )

        return ContextResult(
            context_block      = context_block,
            included_docs      = included,
            excluded_docs      = excluded,
            citation_map       = citation_map,
            tokens_used        = tokens_used,
            tokens_budget      = budget,
            duplicates_removed = n_dupes,
            truncated          = truncated,
        )

    def format_memory(self, memory: List[Message], max_turns: Optional[int] = None) -> str:
        """
        Convert conversation history into a compact text block.
        Returns empty string if no memory.
        """
        max_turns = max_turns or self.max_memory_messages
        recent = memory[-max_turns * 2:]  # each turn = 2 messages (user + assistant)
        if not recent:
            return ""

        lines = []
        for m in recent:
            label = "User" if m.role == "user" else "Assistant"
            lines.append(f"{label}: {m.content.strip()}")
        return "\n".join(lines)

    # ─────────────────────────────────────────
    # Deduplication
    # ─────────────────────────────────────────

    def _deduplicate(self, docs: List[Document]) -> Tuple[List[Document], int]:
        seen_hashes:  set         = set()
        seen_shingles: List[set]  = []
        unique:   List[Document]  = []
        n_dupes = 0

        for doc in docs:
            h = _content_hash(doc.content)
            if h in seen_hashes:
                n_dupes += 1
                continue

            shingles = _shingles(doc.content)
            duplicate = any(
                _jaccard(shingles, prev) >= self.dedup_threshold
                for prev in seen_shingles
            )
            if duplicate:
                n_dupes += 1
                continue

            seen_hashes.add(h)
            seen_shingles.append(shingles)
            unique.append(doc)

        return unique, n_dupes

    # ─────────────────────────────────────────
    # Lost-in-the-middle reordering
    # ─────────────────────────────────────────

    @staticmethod
    def _lost_in_middle_reorder(docs: List[Document]) -> List[Document]:
        """
        Place the most-relevant docs at the beginning and end of the list;
        less-relevant ones go in the middle (where attention is lowest).

        Pattern: [0, 2, 4, ...(odd middle)..., 3, 1]
        """
        n = len(docs)
        result: List[Optional[Document]] = [None] * n
        left, right = 0, n - 1
        for i, doc in enumerate(docs):
            if i % 2 == 0:
                result[left]  = doc
                left  += 1
            else:
                result[right] = doc
                right -= 1
        return [d for d in result if d is not None]

    # ─────────────────────────────────────────
    # Token budget
    # ─────────────────────────────────────────

    def _apply_budget(
        self,
        docs:   List[Document],
        budget: int,
    ) -> Tuple[List[Document], List[Document], bool]:
        included:  List[Document] = []
        excluded:  List[Document] = []
        used = 0
        truncated = False

        for doc in docs:
            doc_tokens = _count_tokens(doc.content)
            if used + doc_tokens <= budget:
                included.append(doc)
                used += doc_tokens
            else:
                # Try partial inclusion (last resort)
                remaining = budget - used
                if remaining > 50:
                    words = doc.content.split()
                    approx_words = int(remaining / 1.3)
                    trimmed = " ".join(words[:approx_words]) + " [...]"
                    partial = Document(
                        content  = trimmed,
                        source   = doc.source,
                        score    = doc.score,
                        metadata = {**doc.metadata, "truncated": True},
                    )
                    included.append(partial)
                    truncated = True
                excluded.append(doc)

        return included, excluded, truncated

    # ─────────────────────────────────────────
    # Formatting
    # ─────────────────────────────────────────

    @staticmethod
    def _format(docs: List[Document]) -> Tuple[str, Dict[int, str]]:
        if not docs:
            return "", {}

        parts:       List[str]       = []
        citation_map: Dict[int, str] = {}

        for i, doc in enumerate(docs, start=1):
            citation_map[i] = doc.source or f"doc_{i}"
            source_label    = f"[{i}] {doc.source}" if doc.source else f"[{i}]"
            score_label     = f" (relevance: {doc.score:.2f})" if doc.score < 1.0 else ""
            header          = f"--- Source {source_label}{score_label} ---"
            parts.append(f"{header}\n{doc.content.strip()}")

        return "\n\n".join(parts), citation_map
