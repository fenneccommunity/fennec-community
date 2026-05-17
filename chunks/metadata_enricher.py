"""
Enriches every chunk with keywords, importance scores, and positional metadata
"""
from __future__ import annotations
import re
import math
import logging
from collections import Counter
from typing import List, Dict

from .doc_model import DocumentChunk, ChunkType

logger = logging.getLogger(__name__)


# Arabic stop words (common subset)
_ARABIC_STOP_WORDS = {
    "من", "في", "على", "إلى", "عن", "مع", "هذا", "هذه", "ذلك", "التي",
    "الذي", "كان", "كانت", "إن", "أن", "لا", "ما", "هو", "هي", "وقد",
    "وكان", "قد", "لم", "لن", "أو", "كل", "بعد", "قبل", "حتى", "عند",
    "ومن", "وفي", "وإلى", "وعلى", "فإن", "فهو", "أيضا", "فقط",
}

# English stop words (common subset)
_ENGLISH_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "on",
    "at", "by", "for", "with", "from", "as", "it", "its", "this", "that",
    "these", "those", "and", "or", "but", "if", "then", "so", "yet",
    "not", "no", "nor", "only", "both", "either", "neither",
}

_ALL_STOP_WORDS = _ARABIC_STOP_WORDS | _ENGLISH_STOP_WORDS


def _tokenize(text: str) -> List[str]:
    """تقسيم النص إلى كلمات بعد تطبيع"""
    return re.findall(r'[\w\u0600-\u06FF]+', text.lower())


class MetadataEnricher:
    """
    Adds to every chunk:
    - keywords: top keywords using simple TF-IDF
    - keyword_density: ratio of meaningful words
    - score: blend of importance, position, and keyword density
    """

    def __init__(
        self,
        max_keywords: int = 10,
        header_score_boost: float = 1.5,
    ) -> None:
        self.max_keywords = max_keywords
        self.header_score_boost = header_score_boost

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def enrich(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Enriches a full list of chunks together (TF-IDF needs the corpus).
        """
        if not chunks:
            return chunks

        # Build corpus TF-IDF data
        corpus_tokens = [_tokenize(c.text) for c in chunks]
        idf = self._compute_idf(corpus_tokens)

        for i, chunk in enumerate(chunks):
            tokens = corpus_tokens[i]
            tf = self._compute_tf(tokens)
            keywords = self._extract_keywords(tf, idf)
            keyword_density = self._keyword_density(tokens)
            score = self._compute_score(chunk, keyword_density, i, len(chunks))

            chunk.metadata.keywords = keywords
            chunk.metadata.keyword_density = keyword_density
            chunk.metadata.score = score

        return chunks

    def enrich_single(self, chunk: DocumentChunk) -> DocumentChunk:
        """إثراء chunk واحد بدون corpus (TF فقط، بدون IDF)"""
        tokens = _tokenize(chunk.text)
        tf = self._compute_tf(tokens)
        # Use TF only (no IDF) for single chunk
        keywords = sorted(
            [(w, s) for w, s in tf.items() if w not in _ALL_STOP_WORDS and len(w) > 2],
            key=lambda x: -x[1]
        )[:self.max_keywords]
        chunk.metadata.keywords = [w for w, _ in keywords]
        chunk.metadata.keyword_density = self._keyword_density(tokens)
        chunk.metadata.score = self._compute_score(chunk, chunk.metadata.keyword_density, 0, 1)
        return chunk

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _compute_tf(self, tokens: List[str]) -> Dict[str, float]:
        if not tokens:
            return {}
        counts = Counter(tokens)
        total = len(tokens)
        return {w: c / total for w, c in counts.items()}

    def _compute_idf(self, corpus: List[List[str]]) -> Dict[str, float]:
        N = len(corpus)
        df: Dict[str, int] = {}
        for tokens in corpus:
            for w in set(tokens):
                df[w] = df.get(w, 0) + 1
        idf: Dict[str, float] = {}
        for w, d in df.items():
            idf[w] = math.log((N + 1) / (d + 1)) + 1.0
        return idf

    def _extract_keywords(
        self,
        tf: Dict[str, float],
        idf: Dict[str, float],
    ) -> List[str]:
        scored = []
        for w, tf_score in tf.items():
            if w in _ALL_STOP_WORDS or len(w) < 2:
                continue
            tfidf = tf_score * idf.get(w, 1.0)
            scored.append((w, tfidf))
        scored.sort(key=lambda x: -x[1])
        return [w for w, _ in scored[: self.max_keywords]]

    def _keyword_density(self, tokens: List[str]) -> float:
        if not tokens:
            return 0.0
        meaningful = [t for t in tokens if t not in _ALL_STOP_WORDS and len(t) > 2]
        return len(meaningful) / len(tokens)

    def _compute_score(
        self,
        chunk: DocumentChunk,
        keyword_density: float,
        position: int,
        total: int,
    ) -> float:
        """
        يحسب درجة أهمية الـ chunk بين 0 و 1.
        Computes chunk importance score between 0 and 1.

        Factors:
        - Keyword density (0.4 weight)
        - Position bonus: first and last chunks are important (0.3)
        - Length normalization: medium-length chunks score higher (0.2)
        - Header bonus (0.1 + boost)
        """
        # Position score: U-shaped (first and last are important)
        if total <= 1:
            position_score = 1.0
        else:
            rel = position / (total - 1)
            position_score = 1 - abs(rel - 0.5) * 2 * 0.5  # 0.5..1.0 range
            position_score = max(0.5, position_score)

        # Length score: prefer chunks between 100-600 chars
        char_count = len(chunk.text)
        if char_count < 50:
            length_score = 0.2
        elif char_count < 100:
            length_score = 0.5
        elif char_count <= 600:
            length_score = 1.0
        else:
            length_score = max(0.5, 1.0 - (char_count - 600) / 2000)

        # Header bonus
        is_header = chunk.metadata.is_header or chunk.metadata.chunk_type == ChunkType.HEADER
        header_bonus = self.header_score_boost if is_header else 1.0

        base_score = (
            keyword_density * 0.4
            + position_score * 0.3
            + length_score * 0.2
            + (0.1 if is_header else 0.0)
        )
        return min(base_score * header_bonus, 1.0)
