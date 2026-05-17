"""
Analyzes query patterns and selects the optimal chunking strategy
"""
from __future__ import annotations
import re
import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


@dataclass
class ChunkingStrategy:
    """Represents a chunking strategy with specific parameters."""
    name: str                           # "faq" | "documentation" | "narrative" | "technical" | "general"
    chunk_size: int                     # الحجم الموصى به
    overlap: int                        # التداخل الموصى به
    use_semantic: bool = False
    use_adaptive: bool = False
    use_structure: bool = False
    description: str = ""


# ────────────────────────────────────────────────────────────────────────
# Predefined strategies
# ────────────────────────────────────────────────────────────────────────

STRATEGIES: Dict[str, ChunkingStrategy] = {
    "faq": ChunkingStrategy(
        name="faq",
        chunk_size=256,
        overlap=32,
        use_semantic=True,
        description="أسئلة وأجوبة — chunks صغيرة ودقيقة"
    ),
    "documentation": ChunkingStrategy(
        name="documentation",
        chunk_size=768,
        overlap=128,
        use_semantic=True,
        use_structure=True,
        description="توثيق تقني — chunks متوسطة مع احترام البنية"
    ),
    "technical": ChunkingStrategy(
        name="technical",
        chunk_size=400,
        overlap=80,
        use_semantic=True,
        use_adaptive=True,
        description="نص تقني كثيف — chunks صغيرة مع adaptive"
    ),
    "narrative": ChunkingStrategy(
        name="narrative",
        chunk_size=1024,
        overlap=200,
        use_semantic=False,
        description="نص سردي — chunks كبيرة للسياق"
    ),
    "general": ChunkingStrategy(
        name="general",
        chunk_size=512,
        overlap=128,
        use_semantic=False,
        description="عام — إعدادات متوازنة"
    ),
}


# ────────────────────────────────────────────────────────────────────────
# Pattern detection
# ────────────────────────────────────────────────────────────────────────

# Patterns → strategy name
_FAQ_PATTERNS = [
    r'\b(FAQ|frequently asked|what is|how to|how do|كيف|ما هو|ما هي|هل يمكن|لماذا)\b',
    r'\?',           # queries with question marks
]
_TECH_PATTERNS = [
    r'\b(API|SDK|function|class|method|algorithm|database|server|config|import|def |class )\b',
    r'```',          # code blocks
    r'\b(pip install|npm install|sudo)\b',
]
_DOC_PATTERNS = [
    r'\b(documentation|manual|guide|reference|overview|introduction|مقدمة|دليل|توثيق)\b',
    r'^#{1,3}\s',    # markdown headers → structured doc
]
_NARRATIVE_PATTERNS = [
    r'\b(story|chapter|novel|narrative|قصة|رواية|فصل|حكاية)\b',
    r'said|replied|whispered|قال|أجاب',
]


class QueryPatternAnalyzer:
    """
    Analyzes user queries and/or document content to suggest the best
    chunking strategy (chunk size, overlap, semantic/adaptive flags).
    """

    def __init__(self, default_strategy: str = "general") -> None:
        if default_strategy not in STRATEGIES:
            raise ValueError(f"Unknown strategy '{default_strategy}'. Choices: {list(STRATEGIES)}")
        self.default_strategy = default_strategy

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def analyze_queries(self, queries: List[str]) -> ChunkingStrategy:
        """
        Suggest a strategy based on a list of representative queries.
        """
        votes: Dict[str, int] = {k: 0 for k in STRATEGIES}
        for q in queries:
            detected = self._classify_text(q)
            votes[detected] += 1
        best = max(votes, key=lambda k: votes[k])
        if votes[best] == 0:
            best = self.default_strategy
        logger.debug(f"[QueryPatternAnalyzer] Query votes: {votes} → '{best}'")
        return STRATEGIES[best]

    def analyze_document(self, text: str) -> ChunkingStrategy:
        """
        Suggest a strategy based on document content.
        """
        detected = self._classify_text(text)
        return STRATEGIES[detected]

    def analyze(
        self,
        text: str,
        sample_queries: Optional[List[str]] = None,
    ) -> ChunkingStrategy:
        """
        Combined document + query analysis.
        """
        doc_strategy = self.analyze_document(text)
        if sample_queries:
            q_strategy = self.analyze_queries(sample_queries)
            # Merge: queries take priority on chunk_size decision
            if q_strategy.name != "general":
                return q_strategy
        return doc_strategy

    def get_strategy(self, name: str) -> ChunkingStrategy:
        """Get a strategy by name, or return default if unknown."""
        return STRATEGIES.get(name, STRATEGIES[self.default_strategy])

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _classify_text(self, text: str) -> str:
        scores: Dict[str, int] = {k: 0 for k in STRATEGIES}

        # FAQ score
        for pat in _FAQ_PATTERNS:
            scores["faq"] += len(re.findall(pat, text, re.IGNORECASE | re.MULTILINE))

        # Technical score
        for pat in _TECH_PATTERNS:
            scores["technical"] += len(re.findall(pat, text, re.IGNORECASE | re.MULTILINE))

        # Documentation score
        for pat in _DOC_PATTERNS:
            scores["documentation"] += len(re.findall(pat, text, re.IGNORECASE | re.MULTILINE))

        # Narrative score
        for pat in _NARRATIVE_PATTERNS:
            scores["narrative"] += len(re.findall(pat, text, re.IGNORECASE | re.MULTILINE))

        best = max(scores, key=lambda k: scores[k])
        if scores[best] == 0:
            return "general"
        return best
