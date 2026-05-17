"""
Backward compatibility shim
"""

from __future__ import annotations

import hashlib
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── validate_chunk (from original context_manager.py) ─────────────────────

def validate_chunk(chunk: object) -> None:
    """Raise TypeError with a descriptive message if chunk is malformed."""
    missing = [
        attr for attr in ("doc_id", "text")
        if not hasattr(chunk, attr)
    ]
    if missing:
        raise TypeError(
            f"Chunk is missing required attributes: {missing}. "
            "Chunk must expose 'doc_id: str' and 'text: str'."
        )
    if not isinstance(chunk.text, str):    # type: ignore
        raise TypeError(f"chunk.text must be str, got {type(chunk.text).__name__}.")
    if not isinstance(chunk.doc_id, str):  # type: ignore
        raise TypeError(f"chunk.doc_id must be str, got {type(chunk.doc_id).__name__}.")


# ── PrefixDeduplication ────────────────────────────────────────────────────

class PrefixDeduplication:
    """Original prefix-based deduplication — kept for backward compatibility."""

    def __init__(self, prefix_length: int = 100):
        self.prefix_length = prefix_length

    def deduplicate(self, chunks: List[Tuple]) -> List[Tuple]:
        seen: set = set()
        unique: List[Tuple] = []
        for chunk, score in chunks:
            if self.prefix_length > len(chunk.text):
                raise ValueError(
                    f"prefix_length {self.prefix_length} exceeds chunk text length {len(chunk.text)}."
                )
            sig = chunk.text[: self.prefix_length].lower()
            if sig not in seen:
                seen.add(sig)
                unique.append((chunk, score))
        return unique


# ── HashDeduplication ──────────────────────────────────────────────────────

class HashDeduplication:
    """Strong SHA-256 hash deduplication — kept for backward compatibility."""

    def deduplicate(self, chunks: List[Tuple]) -> List[Tuple]:
        seen: set = set()
        unique: List[Tuple] = []
        for chunk, score in chunks:
            normalised = " ".join(chunk.text.lower().split())
            sig = hashlib.sha256(normalised.encode()).hexdigest()
            if sig not in seen:
                seen.add(sig)
                unique.append((chunk, score))
        return unique


# ── ScoreRanking ───────────────────────────────────────────────────────────

class ScoreRanking:
    """Sort by similarity score, highest first — kept for backward compatibility."""

    def rank(self, chunks: List[Tuple], max_chunks: Optional[int]) -> List[Tuple]:
        ranked = sorted(chunks, key=lambda x: x[1], reverse=True)
        return ranked[:max_chunks] if max_chunks is not None else ranked


# ── GreedyCompression ──────────────────────────────────────────────────────

class GreedyCompression:
    """Greedy context compression — kept for backward compatibility."""

    MIN_CONTENT_LENGTH = 50

    def compress(
        self,
        context:   str,
        target:    int,
        counter,
        separator: str,
    ) -> str:
        if counter(context) <= target:
            return context
        if target <= 0:
            return ""

        parts = context.split(separator)
        if len(parts) <= 2:
            return self._truncate(context, target, counter)

        header, *middle, footer = parts
        overhead  = counter(header) + counter(footer) + 2 * counter(separator)
        available = target - overhead

        if available < self.MIN_CONTENT_LENGTH:
            return self._truncate(header, target, counter)

        compressed_middle: List[str] = []
        used = 0
        for part in middle:
            part_len  = counter(part)
            remaining = available - used
            if remaining <= 0:
                break
            if used + part_len <= available:
                compressed_middle.append(part)
                used += part_len
            elif remaining > self.MIN_CONTENT_LENGTH:
                compressed_middle.append(self._truncate(part, remaining, counter))
                break

        result = separator.join([header, *compressed_middle, footer])
        if counter(result) > target:
            result = self._truncate(result, target, counter)
        return result

    @staticmethod
    def _truncate(text: str, max_len: int, counter) -> str:
        if counter(text) <= max_len:
            return text
        lo, hi = 0, len(text)
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if counter(text[:mid]) <= max_len - 3:
                lo = mid
            else:
                hi = mid
        truncated  = text[:lo]
        last_space = truncated.rfind(" ")
        if last_space > lo * 0.8:
            truncated = truncated[:last_space]
        return truncated + "..."
