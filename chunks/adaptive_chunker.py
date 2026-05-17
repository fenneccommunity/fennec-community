"""
AdaptiveChunker — تقسيم ديناميكي يعدّل حجم الـ chunk حسب كثافة المعلومات
Dynamically adjusts chunk size based on information density and text type
"""
from __future__ import annotations
import re
import uuid
import logging
from typing import List, Optional

from .base import BaseChunker
from .doc_model import DocumentChunk, ChunkMetadata, ChunkType
from .chunk_config import ChunkConfig

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────
# Information Density Analyzer
# ────────────────────────────────────────────────────────────────────────

# Indicators of technical / high-density text
_TECHNICAL_INDICATORS = re.compile(
    r'('
    r'\b(def|class|import|return|function|var|const|let|if|else|for|while|try|catch'
    r'|SELECT|FROM|WHERE|JOIN|INSERT|UPDATE|DELETE'
    r'|API|SDK|HTTP|JSON|XML|SQL|REST|GraphQL'
    r'|algorithm|complexity|O\(n\)|theorem|lemma|proof'
    r'|equation|formula|integral|derivative'
    r')\b'
    r'|[{}<>\[\]();=+\-*/\\]'      # code punctuation
    r'|\b\d+\.\d+\b'               # decimal numbers
    r')',
    re.IGNORECASE | re.MULTILINE,
)

# Arabic complex vocabulary / formal register markers
_ARABIC_COMPLEX = re.compile(
    r'[\u0600-\u06FF]{7,}',   # long Arabic words (likely technical/formal)
)


def _estimate_density(text: str) -> float:
    """
    Returns information density estimate between 0 and 1.
    Higher = more technical / dense.
    """
    if not text:
        return 0.0
    total_chars = len(text)
    tech_hits = len(_TECHNICAL_INDICATORS.findall(text))
    arabic_complex_hits = len(_ARABIC_COMPLEX.findall(text))

    # Unique vocab ratio (high ratio → informational)
    words = re.findall(r'\w+', text.lower())
    unique_ratio = len(set(words)) / max(len(words), 1)

    # Average word length (longer words → denser / technical)
    avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
    word_len_score = min(avg_word_len / 10.0, 1.0)

    # Weighted sum
    density = (
        min(tech_hits / max(total_chars / 100, 1), 1.0) * 0.4
        + unique_ratio * 0.3
        + word_len_score * 0.2
        + min(arabic_complex_hits / max(total_chars / 200, 1), 1.0) * 0.1
    )
    return min(density, 1.0)


class AdaptiveChunker(BaseChunker):
    """
    Splits text with dynamic sizing adapting to information density:
    - Dense technical text → smaller chunks
    - Narrative / simple text → larger chunks
    """

    def __init__(
        self,
        config: Optional[ChunkConfig] = None,
        *,
        base_size: int = 512,
        min_size: int = 100,
        max_size: int = 1500,
        overlap: int = 128,
        technical_threshold: float = 0.5,
        language: str = "auto",
    ) -> None:
        cfg = config or ChunkConfig()
        self.base_size = base_size if config is None else cfg.adaptive_base_size
        self.min_size = min_size if config is None else cfg.adaptive_min_size
        self.max_size = max_size if config is None else cfg.adaptive_max_size
        self.overlap = overlap if config is None else cfg.overlap
        self.technical_threshold = technical_threshold if config is None else cfg.adaptive_technical_threshold
        self.language = language

    # ------------------------------------------------------------------ #
    # BaseChunker implementation                                           #
    # ------------------------------------------------------------------ #

    def _chunk_impl(self, text: str, doc_id: str, source: str) -> List[DocumentChunk]:
        if not text or not text.strip():
            return []

        # Estimate density for the full text first
        density = _estimate_density(text)
        target_size = self._compute_target_size(density)
        logger.debug(f"[AdaptiveChunker] density={density:.3f} → target_size={target_size}")

        # Split into sentences for granular control
        sentences = self._split_sentences(text)
        chunks: List[DocumentChunk] = []
        current_sentences: List[str] = []
        current_len = 0
        char_cursor = 0
        position = 0

        for sentence in sentences:
            sentence_len = len(sentence)
            local_density = _estimate_density(sentence)
            local_target = self._compute_target_size(local_density)

            if current_len + sentence_len > local_target and current_sentences:
                chunk_text = " ".join(current_sentences).strip()
                if len(chunk_text) >= (config := ChunkConfig()).min_chunk_size:
                    chunks.append(self._make_chunk(
                        chunk_text, doc_id=doc_id, source=source,
                        position=position,
                        char_start=char_cursor - current_len,
                        density=density,
                    ))
                    position += 1

                # Overlap: keep last sentences up to self.overlap chars
                overlap_buf: List[str] = []
                overlap_len = 0
                for s in reversed(current_sentences):
                    if overlap_len + len(s) <= self.overlap:
                        overlap_buf.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break
                current_sentences = overlap_buf
                current_len = overlap_len

            current_sentences.append(sentence)
            current_len += sentence_len
            char_cursor += sentence_len + 1

        # Flush remaining
        if current_sentences:
            chunk_text = " ".join(current_sentences).strip()
            if chunk_text:
                chunks.append(self._make_chunk(
                    chunk_text, doc_id=doc_id, source=source,
                    position=position,
                    char_start=char_cursor - current_len,
                    density=density,
                ))

        return chunks

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _compute_target_size(self, density: float) -> int:
        """
        حجم الـ chunk المستهدف بناءً على كثافة المعلومات.
        High density → min_size … Low density → max_size
        """
        # Linear interpolation between max and min based on density
        size = self.max_size - density * (self.max_size - self.min_size)
        return max(self.min_size, min(int(size), self.max_size))

    def _split_sentences(self, text: str) -> List[str]:
        """تقسيم إلى جمل باستخدام punctuation"""
        parts = re.split(r'(?<=[.!?؟\u06D4])\s+|\n{2,}', text)
        return [p.strip() for p in parts if p.strip()]

    def _make_chunk(
        self,
        text: str,
        doc_id: str,
        source: str,
        position: int,
        char_start: int = 0,
        density: float = 0.0,
    ) -> DocumentChunk:
        return DocumentChunk(
            text=text,
            chunk_id=str(uuid.uuid4()),
            doc_id=doc_id,
            metadata=ChunkMetadata(
                source=source,
                position=position,
                chunk_type=ChunkType.ADAPTIVE,
                language=self.language,
                char_start=char_start,
                char_end=char_start + len(text),
                extra={"density": round(density, 4)},
            ),
        )
