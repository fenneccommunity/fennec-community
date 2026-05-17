"""
query_analyzer.py — Intelligent Query Understanding Module
محلل الاستعلامات الذكي — يستخرج النية والتعقيد والكلمات المفتاحية
"""

from __future__ import annotations

import re
import logging
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

from .config import ContextEngineConfig
from .models import QueryAnalysis, QueryComplexity, QueryIntent
from .strategy import QueryAnalyzerStrategy

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Linguistic Signals
# ─────────────────────────────────────────────────────────────────────────────

_INTENT_SIGNALS: Dict[QueryIntent, Dict[str, List[str]]] = {
    QueryIntent.FACTUAL: {
        "ar": ["ما", "من", "متى", "أين", "كم", "ما هو", "ما هي", "ما اسم", "من هو", "من هي"],
        "en": ["what", "who", "when", "where", "which", "how many", "how much"],
    },
    QueryIntent.REASONING: {
        "ar": ["لماذا", "كيف", "ما سبب", "ما علة", "فسّر", "وضّح", "اشرح"],
        "en": ["why", "how", "explain", "describe", "elaborate", "analyze", "reason"],
    },
    QueryIntent.COMPARATIVE: {
        "ar": ["قارن", "الفرق", "أيهما", "أفضل", "مقارنة", "مقابل", "بين"],
        "en": ["compare", "difference", "versus", "vs", "better", "worse", "contrast"],
    },
    QueryIntent.PROCEDURAL: {
        "ar": ["كيفية", "خطوات", "طريقة", "كيف أفعل", "كيف يمكن", "آلية"],
        "en": ["how to", "steps", "procedure", "process", "guide", "tutorial", "implement"],
    },
    QueryIntent.CODE: {
        "ar": ["كود", "برمجة", "دالة", "كلاس", "مثال كود", "اكتب كود", "برمج"],
        "en": ["code", "function", "class", "implement", "script", "program", "snippet"],
    },
    QueryIntent.SUMMARIZE: {
        "ar": ["لخّص", "ملخص", "باختصار", "أعطني فكرة", "شرح مختصر", "ما هي النقاط"],
        "en": ["summarize", "summary", "overview", "briefly", "tldr", "main points"],
    },
}

_COMPLEXITY_HIGH_SIGNALS = re.compile(
    r"\b(analyze|compare|evaluate|critique|pros.and.cons|trade.?off"
    r"|قارن|حلل|قيّم|عوامل|مزايا وعيوب|الفرق بين)\b",
    re.I,
)
_COMPLEXITY_LOW_SIGNALS = re.compile(
    r"^(ما|من|متى|أين|what is|who is|when|where)\s+\w+",
    re.I,
)

_STOP_WORDS_AR: Set[str] = {
    "في", "من", "إلى", "على", "عن", "مع", "هذا", "هذه", "ذلك", "التي", "الذي",
    "أن", "إن", "كان", "كانت", "هو", "هي", "هم", "نحن", "أنت", "ما", "لا",
    "لم", "قد", "كل", "بعض", "بين", "حيث", "ثم", "أو", "لكن", "و", "ب", "ل",
}
_STOP_WORDS_EN: Set[str] = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "is", "are", "was", "were", "be", "been", "have",
    "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "this", "that", "these", "those", "it", "its", "we", "you", "he", "she",
    "they", "them", "their", "our", "your", "my", "i", "not", "can", "get",
}

_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
_ENTITY_RE = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*"        # English proper nouns
    r"|[\u0600-\u06FF]{3,}(?:\s+[\u0600-\u06FF]{3,}){1,3})\b"  # Arabic multi-word
)


# ─────────────────────────────────────────────────────────────────────────────
#  Default QueryAnalyzer Implementation
# ─────────────────────────────────────────────────────────────────────────────

class RuleBasedQueryAnalyzer(QueryAnalyzerStrategy):
    """
   query analyzer that uses rule-based heuristics to extract intent, complexity, keywords, and entities from the query.
   
    """

    def __init__(self, config: ContextEngineConfig):
        self.cfg = config.query_analyzer

    def analyze(self, query: str) -> QueryAnalysis:
        if not query or not query.strip():
            return self._empty_analysis(query)

        normalized = self._normalize(query)
        is_arabic  = self._detect_arabic(normalized)
        lang       = "ar" if is_arabic else "en"

        intent     = self._extract_intent(normalized, lang)
        complexity = self._estimate_complexity(normalized, lang)
        keywords   = self._extract_keywords(normalized, lang)
        entities   = self._extract_entities(normalized)
        q_type     = self._question_type(normalized, lang)
        budget     = self._compute_budget(complexity)

        analysis = QueryAnalysis(
            original_query = query,
            normalized     = normalized,
            intent         = intent,
            complexity     = complexity,
            keywords       = keywords,
            entities       = entities,
            is_arabic      = is_arabic,
            language       = lang,
            question_type  = q_type,
            context_budget = budget,
        )

        logger.debug(
            "QueryAnalyzer | intent=%s | complexity=%s | keywords=%s",
            intent.value, complexity.value, keywords[:5],
        )
        return analysis

    # ── Private helpers ───────────────────────────────────────────────────

    def _normalize(self, query: str) -> str:
        """تطبيع النص: إزالة التشكيل، توحيد المسافات"""
        text = query.strip()
        text = re.sub(r"[\u064B-\u0652]", "", text)          # تشكيل
        text = re.sub(r"[إأآا]", "ا", text)                  # توحيد الألف
        text = re.sub(r"\s+", " ", text)
        return text

    def _detect_arabic(self, text: str) -> bool:
        arabic_chars = len(_ARABIC_RE.findall(text))
        total_chars  = max(len(text.replace(" ", "")), 1)
        return (arabic_chars / total_chars) >= self.cfg.arabic_threshold

    def _extract_intent(self, text: str, lang: str) -> QueryIntent:
        text_lower = text.lower()
        scores: Dict[QueryIntent, int] = {}

        for intent, signals in _INTENT_SIGNALS.items():
            lang_signals = signals.get(lang, []) + signals.get(
                "ar" if lang == "en" else "en", []
            )
            count = sum(1 for s in lang_signals if s.lower() in text_lower)
            if count > 0:
                scores[intent] = count

        if not scores:
            return QueryIntent.FACTUAL
        return max(scores, key=scores.get)

    def _estimate_complexity(self, text: str, lang: str) -> QueryComplexity:
        word_count = len(text.split())

        if _COMPLEXITY_HIGH_SIGNALS.search(text):
            return QueryComplexity.HIGH
        if word_count > 20:
            return QueryComplexity.HIGH
        if _COMPLEXITY_LOW_SIGNALS.match(text) and word_count <= 7:
            return QueryComplexity.LOW
        if word_count <= 12:
            return QueryComplexity.LOW
        return QueryComplexity.MEDIUM

    def _extract_keywords(self, text: str, lang: str) -> List[str]:
        stop = _STOP_WORDS_AR if lang == "ar" else _STOP_WORDS_EN
        words = re.findall(r"[\w\u0600-\u06FF]{2,}", text.lower())
        words = [w for w in words if w not in stop]
        freq  = Counter(words)
        # Boost longer words (more specific)
        scored = [(w, cnt * (1 + len(w) * 0.05)) for w, cnt in freq.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [w for w, _ in scored[:12]]

    def _extract_entities(self, text: str) -> List[str]:
        entities = [m.group(0).strip() for m in _ENTITY_RE.finditer(text)]
        # Add custom entities from config
        for ent in self.cfg.custom_entities:
            if ent.lower() in text.lower():
                entities.append(ent)
        return list(dict.fromkeys(entities))[:8]  # unique, keep order

    def _question_type(self, text: str, lang: str) -> str:
        text_l = text.lower().strip()
        wh_ar  = ["ما", "من", "متى", "أين", "كيف", "لماذا", "كم"]
        wh_en  = ["what", "who", "when", "where", "how", "why", "which"]
        yn_ar  = ["هل", "أ"]
        yn_en  = ["is", "are", "was", "were", "do", "does", "did", "can", "could", "will"]

        if lang == "ar":
            if any(text_l.startswith(w) for w in wh_ar):
                return "wh-question"
            if any(text_l.startswith(w) for w in yn_ar):
                return "yes/no"
        else:
            if any(text_l.startswith(w) for w in wh_en):
                return "wh-question"
            if any(text_l.startswith(w) for w in yn_en):
                return "yes/no"
        return "open"

    def _compute_budget(self, complexity: QueryComplexity) -> int:
        mapping = {
            QueryComplexity.LOW:    self.cfg.budget_low,
            QueryComplexity.MEDIUM: self.cfg.budget_medium,
            QueryComplexity.HIGH:   self.cfg.budget_high,
        }
        return mapping[complexity]

    def _empty_analysis(self, query: str) -> QueryAnalysis:
        return QueryAnalysis(
            original_query = query,
            normalized     = "",
            intent         = QueryIntent.UNKNOWN,
            complexity     = QueryComplexity.LOW,
            context_budget = self.cfg.budget_low,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_query_analyzer(
    config:   ContextEngineConfig,
    strategy: Optional[QueryAnalyzerStrategy] = None,
) -> QueryAnalyzerStrategy:
    """
Factory function to create a QueryAnalyzerStrategy instance based on the provided configuration and optional custom strategy.
    """
    return strategy or RuleBasedQueryAnalyzer(config)
