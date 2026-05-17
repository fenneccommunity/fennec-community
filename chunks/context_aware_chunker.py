"""
Preserves context using window-based sentence merging to avoid cutting ideas mid-thought
"""
from __future__ import annotations
import re
import uuid
import logging
from typing import List, Optional

from .base import BaseChunker
from .doc_model import DocumentChunk, ChunkMetadata, ChunkType, DocumentType
from .chunk_config import ChunkConfig

logger = logging.getLogger(__name__)


class ContextAwareChunker(BaseChunker):
    """
    Ensures every chunk carries enough context:
    1. Split into sentences.
    2. Group sentences in overlapping sliding windows.
    3. Each window → chunk with overlap for context.
    4. Never cuts in the middle of a sentence.
    """

    def __init__(
        self,
        config: Optional[ChunkConfig] = None,
        *,
        chunk_size: int = 512,
        overlap: int = 128,
        window_size: int = 2,           # extra sentences added on each side
        min_chunk_size: int = 50,
        language: str = "auto",
    ) -> None:
        cfg = config or ChunkConfig()
        self.chunk_size = chunk_size if config is None else cfg.chunk_size
        self.overlap = overlap if config is None else cfg.overlap
        self.window_size = window_size if config is None else cfg.context_window_size
        self.min_chunk_size = min_chunk_size if config is None else cfg.min_chunk_size
        self.language = language

    # ------------------------------------------------------------------ #
    # BaseChunker implementation                                           #
    # ------------------------------------------------------------------ #

    def _chunk_impl(self, text: str, doc_id: str, source: str) -> List[DocumentChunk]:
        if not text or not text.strip():
            return []

        sentences = self._split_sentences(text)
        if not sentences:
            return []

        # Group sentences into primary groups by size limit
        primary_groups = self._group_sentences(sentences)

        # Apply window context: add surrounding sentences to each group
        windowed_groups = self._apply_window(primary_groups, sentences)

        chunks: List[DocumentChunk] = []
        char_cursor = 0
        for pos, group_text in enumerate(windowed_groups):
            group_text = group_text.strip()
            if len(group_text) >= self.min_chunk_size:
                chunks.append(self._make_chunk(
                    group_text, doc_id=doc_id, source=source,
                    position=pos,
                    char_start=char_cursor,
                ))
                char_cursor += len(group_text)
        return chunks

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _split_sentences(self, text: str) -> List[str]:
        """تقسيم النص إلى جمل مع دعم العربية"""
        parts = re.split(r'(?<=[.!?؟\u06D4،])\s+|\n{2,}', text)
        return [p.strip() for p in parts if p.strip()]

    def _group_sentences(self, sentences: List[str]) -> List[List[int]]:
        """
        تجميع الجمل في مجموعات لا تتجاوز chunk_size.
        Returns list of sentence index groups.
        """
        groups: List[List[int]] = []
        current: List[int] = []
        current_len = 0

        for i, sent in enumerate(sentences):
            sent_len = len(sent)
            if current_len + sent_len > self.chunk_size and current:
                groups.append(current)
                # Keep overlap sentences
                overlap_sentences: List[int] = []
                overlap_len = 0
                for prev_idx in reversed(current):
                    if overlap_len + len(sentences[prev_idx]) <= self.overlap:
                        overlap_sentences.insert(0, prev_idx)
                        overlap_len += len(sentences[prev_idx])
                    else:
                        break
                current = overlap_sentences
                current_len = overlap_len
            current.append(i)
            current_len += sent_len

        if current:
            groups.append(current)
        return groups

    def _apply_window(
        self,
        groups: List[List[int]],
        sentences: List[str],
    ) -> List[str]:
        """
        لكل مجموعة، أضف window_size جمل قبلها وبعدها للحفاظ على السياق.
        For each group, prepend/append window_size sentences for context.
        """
        result: List[str] = []
        n = len(sentences)

        for group in groups:
            if not group:
                continue
            first_idx = group[0]
            last_idx = group[-1]

            # Window before
            context_before_start = max(0, first_idx - self.window_size)
            # Window after
            context_after_end = min(n - 1, last_idx + self.window_size)

            all_indices = list(range(context_before_start, context_after_end + 1))
            chunk_text = " ".join(sentences[i] for i in all_indices)
            result.append(chunk_text)

        return result

    def _make_chunk(
        self,
        text: str,
        doc_id: str,
        source: str,
        position: int,
        char_start: int = 0,
    ) -> DocumentChunk:
        return DocumentChunk(
            text=text,
            chunk_id=str(uuid.uuid4()),
            doc_id=doc_id,
            metadata=ChunkMetadata(
                source=source,
                position=position,
                chunk_type=ChunkType.WINDOW,
                language=self.language,
                char_start=char_start,
                char_end=char_start + len(text),
            ),
        )
