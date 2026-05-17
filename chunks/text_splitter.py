"""
Rule-based text splitters, backward-compatible with existing codebase + Arabic support
"""
from __future__ import annotations
import re
from typing import List, Optional, Callable

from .base import TextSplitter
from .chunk_config import ChunkConfig

config = ChunkConfig()


class CharacterTextSplitter(TextSplitter):
    """
    Character / string separator-based text splitter.
    """

    def __init__(self, separator: str = "\n\n", **kwargs) -> None:
        super().__init__(**kwargs)
        self.separator = separator

    def split_text(self, text: str) -> List[str]:
        splits = text.split(self.separator) if self.separator else list(text)
        return self._merge_splits(splits)

    def _merge_splits(self, splits: List[str]) -> List[str]:
        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        for split in splits:
            split_len = self.length_function(split)
            if current_len + split_len > self.chunk_size and current:
                chunks.append(self.separator.join(current))
                # Overlap
                overlap_buf: List[str] = []
                overlap_len = 0
                for prev in reversed(current):
                    if overlap_len + self.length_function(prev) <= self.chunk_overlap:
                        overlap_buf.insert(0, prev)
                        overlap_len += self.length_function(prev)
                    else:
                        break
                current = overlap_buf
                current_len = overlap_len

            current.append(split)
            current_len += split_len

        if current:
            chunks.append(self.separator.join(current))
        return [c for c in chunks if c.strip()]


class RecursiveCharacterTextSplitter(TextSplitter):
    """
    Hierarchical splitting: tries multiple separators from most to least specific.
    Includes Arabic-aware separators.
    """

    # Default separators include Arabic sentence terminators
    DEFAULT_SEPARATORS = ["\n\n", "\n", ".", "؟", "!", "،", "؛", " ", ""]
    ARABIC_SEPARATORS  = [".\n", "\n\n", "\n", ".", "؟", "!", "،", "؛", " ", ""]

    def __init__(
        self,
        separators: Optional[List[str]] = None,
        arabic_mode: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if separators is not None:
            self.separators = separators
        elif arabic_mode:
            self.separators = self.ARABIC_SEPARATORS
        else:
            self.separators = self.DEFAULT_SEPARATORS

    def split_text(self, text: str) -> List[str]:
        return self._split_recursive(text, self.separators)

    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        final_chunks: List[str] = []

        # Find first separator that exists in text
        separator = separators[-1]
        for sep in separators:
            if sep == "" or sep in text:
                separator = sep
                break

        splits = text.split(separator) if separator else list(text)
        good_splits: List[str] = []

        for split in splits:
            if self.length_function(split) < self.chunk_size:
                good_splits.append(split)
            else:
                if good_splits:
                    final_chunks.extend(self._merge_splits(good_splits, separator))
                    good_splits = []
                # Recurse with remaining separators
                remaining = [s for s in separators if s != separator] or [""]
                final_chunks.extend(self._split_recursive(split, remaining))

        if good_splits:
            final_chunks.extend(self._merge_splits(good_splits, separator))

        return final_chunks

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        for split in splits:
            split_len = self.length_function(split)
            if current_len + split_len > self.chunk_size and current:
                text = separator.join(current)
                if text.strip():
                    chunks.append(text)
                # Overlap
                overlap_buf: List[str] = []
                overlap_len = 0
                for prev in reversed(current):
                    if overlap_len + self.length_function(prev) <= self.chunk_overlap:
                        overlap_buf.insert(0, prev)
                        overlap_len += self.length_function(prev)
                    else:
                        break
                current = overlap_buf
                current_len = overlap_len

            current.append(split)
            current_len += split_len

        if current:
            text = separator.join(current)
            if text.strip():
                chunks.append(text)
        return chunks


class TokenTextSplitter(TextSplitter):
    """
    Token-count-based splitter using tiktoken.
    """

    def __init__(self, encoding_name: str = config.model_token, **kwargs) -> None:
        super().__init__(**kwargs)
        try:
            import tiktoken
            self.tokenizer = tiktoken.get_encoding(encoding_name)
        except ImportError:
            raise ImportError("Please install tiktoken: pip install tiktoken")

    def split_text(self, text: str) -> List[str]:
        tokens = self.tokenizer.encode(text)
        chunks: List[str] = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i : i + self.chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            if chunk_text.strip():
                chunks.append(chunk_text)
        return chunks


class SentenceTextSplitter(TextSplitter):
    """
    Sentence-boundary-respecting splitter — useful for Arabic and context preservation.
    """

    # Pattern for sentence boundaries — supports Arabic
    _SENT_BOUNDARY = re.compile(
        r'(?<=[.!?؟\u06D4।।])\s+|(?<=\n)\n+'
    )

    def split_text(self, text: str) -> List[str]:
        sentences = [s.strip() for s in self._SENT_BOUNDARY.split(text) if s.strip()]
        if not sentences:
            return [text] if text.strip() else []

        chunks: List[str] = []
        current_sentences: List[str] = []
        current_len = 0

        for sent in sentences:
            sent_len = self.length_function(sent)
            if current_len + sent_len > self.chunk_size and current_sentences:
                chunks.append(" ".join(current_sentences))
                # Overlap: keep last N chars
                overlap_buf: List[str] = []
                overlap_len = 0
                for s in reversed(current_sentences):
                    if overlap_len + self.length_function(s) <= self.chunk_overlap:
                        overlap_buf.insert(0, s)
                        overlap_len += self.length_function(s)
                    else:
                        break
                current_sentences = overlap_buf
                current_len = overlap_len

            current_sentences.append(sent)
            current_len += sent_len

        if current_sentences:
            chunks.append(" ".join(current_sentences))
        return [c for c in chunks if c.strip()]
