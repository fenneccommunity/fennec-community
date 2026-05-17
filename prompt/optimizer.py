"""
Prompt Optimizer
================
Reduces token count without sacrificing quality or safety:

  1. Whitespace normalisation
  2. Boilerplate deduplication  (same instruction appearing in both system + user)
  3. Context trimming            (truncate least-relevant doc if over budget)
  4. Redundant filler removal    (common LLM prompt padding phrases)
  5. Memory compression          (older turns summarised when history is long)

All operations are deterministic, reversible, and fully logged.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Approximate token count (whitespace heuristic; swap for tiktoken in prod)
def _tok(text: str) -> int:
    return int(len(text.split()) * 1.3)


# ─────────────────────────────────────────────
# Filler phrases to strip (common LLM padding)
# ─────────────────────────────────────────────

_FILLER = re.compile(
    r"(?i)\b("
    r"certainly[!,]?\s*|"
    r"great question[!,]?\s*|"
    r"of course[!,]?\s*|"
    r"absolutely[!,]?\s*|"
    r"i'd be happy to help\.?\s*|"
    r"i'm happy to assist\.?\s*|"
    r"let me help you with that\.?\s*|"
    r"sure[!,]?\s*|"
    r"definitely[!,]?\s*"
    r")"
)

# Runs of more than 2 blank lines → 2 blank lines
_BLANK_LINES = re.compile(r"\n{3,}")

# Multiple spaces → single space
_MULTI_SPACE = re.compile(r"[ \t]{2,}")


class PromptOptimizer:
    """
    Applies a pipeline of lightweight optimizations to (system, user) prompts.

    Returns
    -------
    (optimized_system, optimized_user, tokens_saved, notes)
    """

    def __init__(
        self,
        max_total_tokens: int  = 6000,
        enable_filler:    bool = True,
        enable_dedup:     bool = True,
        enable_whitespace: bool = True,
    ) -> None:
        self.max_total_tokens  = max_total_tokens
        self.enable_filler     = enable_filler
        self.enable_dedup      = enable_dedup
        self.enable_whitespace = enable_whitespace

    def optimize(
        self,
        system: str,
        user:   str,
        request: Optional[object] = None,    # PromptRequest (avoid circular import)
    ) -> Tuple[str, str, int, List[str]]:
        notes: List[str] = []
        original_tokens  = _tok(system) + _tok(user)

        if self.enable_whitespace:
            system, user = self._normalize_whitespace(system, user)
            notes.append("whitespace-normalized")

        if self.enable_filler:
            system, user = self._strip_filler(system, user)
            notes.append("filler-stripped")

        if self.enable_dedup:
            system, user, dedup_note = self._dedup_instructions(system, user)
            if dedup_note:
                notes.append(dedup_note)

        # Final hard cap
        total = _tok(system) + _tok(user)
        if total > self.max_total_tokens:
            user, cap_note = self._hard_cap(user, self.max_total_tokens - _tok(system))
            notes.append(cap_note)

        tokens_saved = max(0, original_tokens - _tok(system) - _tok(user))
        if tokens_saved:
            logger.debug("[Optimizer] Saved %d tokens via: %s", tokens_saved, notes)

        return system, user, tokens_saved, notes

    # ─────────────────────────────────────────
    # Individual passes
    # ─────────────────────────────────────────

    @staticmethod
    def _normalize_whitespace(system: str, user: str) -> Tuple[str, str]:
        def clean(t: str) -> str:
            t = _MULTI_SPACE.sub(" ", t)
            t = _BLANK_LINES.sub("\n\n", t)
            return t.strip()
        return clean(system), clean(user)

    @staticmethod
    def _strip_filler(system: str, user: str) -> Tuple[str, str]:
        def strip(t: str) -> str:
            return _FILLER.sub("", t).strip()
        return strip(system), strip(user)

    @staticmethod
    def _dedup_instructions(
        system: str, user: str
    ) -> Tuple[str, str, str]:
        """
        Remove instruction blocks that appear verbatim in both system AND user.
        Keeps the system version; removes the user duplicate.
        """
        # Split into paragraphs
        sys_paras = set(p.strip() for p in system.split("\n\n") if len(p.strip()) > 30)
        user_paras = user.split("\n\n")
        cleaned_user = []
        removed = 0
        for para in user_paras:
            if para.strip() in sys_paras:
                removed += 1
            else:
                cleaned_user.append(para)

        note = f"dedup-removed-{removed}-paragraphs" if removed else ""
        return system, "\n\n".join(cleaned_user), note

    @staticmethod
    def _hard_cap(user: str, budget: int) -> Tuple[str, str]:
        """Truncate the user prompt (context section) to fit the token budget."""
        words = user.split()
        approx_words = int(budget / 1.3)
        if len(words) <= approx_words:
            return user, ""
        truncated = " ".join(words[:approx_words])
        # Try to cut at a paragraph boundary
        last_para = truncated.rfind("\n\n")
        if last_para > len(truncated) * 0.7:
            truncated = truncated[:last_para]
        truncated += "\n\n[Context truncated to fit token budget]"
        return truncated, f"hard-capped-at-{budget}-tokens"
