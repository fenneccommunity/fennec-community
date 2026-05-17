"""
Central orchestrator combining all intelligent chunking components
"""
from __future__ import annotations
import logging
import uuid
from typing import List, Optional, Dict, Any
from enum import Enum

from .doc_model import DocumentChunk, Document
from .chunk_config import ChunkConfig
from .semantic_chunker import SemanticChunker
from .adaptive_chunker import AdaptiveChunker
from .structure_aware_chunker import StructureAwareChunker
from .context_aware_chunker import ContextAwareChunker
from .metadata_enricher import MetadataEnricher
from .deduplicator import Deduplicator
from .query_pattern_analyzer import QueryPatternAnalyzer, ChunkingStrategy as QStrategy

logger = logging.getLogger(__name__)


class ChunkMode(str, Enum):
    """
    وضع التقسيم
    Chunking mode selection
    """
    AUTO        = "auto"          # يختار ChunkManager تلقائياً
    SEMANTIC    = "semantic"      # SemanticChunker فقط
    ADAPTIVE    = "adaptive"      # AdaptiveChunker فقط
    STRUCTURAL  = "structural"    # StructureAwareChunker فقط
    CONTEXTUAL  = "contextual"    # ContextAwareChunker فقط
    HYBRID      = "hybrid"        # semantic + structural معاً


class ChunkManager:
    """
    ===================================================================
    AI-Powered Intelligent Chunking System — Production-Grade
    ===================================================================


    Manages the full chunking lifecycle:
    1. Query-aware strategy selection (optional)
    2. Document structure parsing
    3. Chunking (semantic / adaptive / structural / contextual / hybrid)
    4. Metadata enrichment (keywords, scores)
    5. Deduplication
    6. Finalization (position, total_chunks)

    Usage:
        manager = ChunkManager(config=ChunkConfig(use_semantic_chunking=True))
        chunks = manager.process(text, source="my_doc.pdf")
    """

    def __init__(
        self,
        config: Optional[ChunkConfig] = None,
        mode: ChunkMode = ChunkMode.AUTO,
        *,
        device: Optional[str] = None,
        language: str = "auto",
    ) -> None:
        self.config = config or ChunkConfig()
        self.mode = mode
        self.device = device
        self.language = language

        # ── Chunkers ──────────────────────────────────────────────────
        self._semantic = SemanticChunker(
            config=self.config,
            language=language,
            device=device,
        )
        self._adaptive = AdaptiveChunker(
            config=self.config,
            language=language,
        )
        self._structural = StructureAwareChunker(
            config=self.config,
            language=language,
        )
        self._contextual = ContextAwareChunker(
            config=self.config,
            language=language,
        )

        # ── Post-processors ───────────────────────────────────────────
        self._enricher = MetadataEnricher(
            max_keywords=self.config.keyword_max_count,
            header_score_boost=self.config.header_score_boost,
        )
        self._dedup = Deduplicator(
            use_hash=self.config.dedup_use_hash,
            use_similarity=False,          # similarity dedup only on explicit request
            similarity_threshold=self.config.dedup_similarity_threshold,
        )

        # ── Query-aware analyzer ──────────────────────────────────────
        self._query_analyzer = QueryPatternAnalyzer(
            default_strategy=self.config.default_query_pattern
        )

    # ------------------------------------------------------------------ #
    # Primary public API                                                   #
    # ------------------------------------------------------------------ #

    def process(
        self,
        text: str,
        doc_id: Optional[str] = None,
        source: str = "",
        *,
        sample_queries: Optional[List[str]] = None,
        mode_override: Optional[ChunkMode] = None,
    ) -> List[DocumentChunk]:
        """
        المدخل الرئيسي: نص → قائمة chunks مُثراة.
        Main entry point: text → enriched, deduplicated chunk list.

        Args:
            text:           النص المراد تقسيمه
            doc_id:         معرف المستند (يُنشأ تلقائياً إن لم يُعطَ)
            source:         مصدر المستند (اسم الملف، URL، ...)
            sample_queries: أسئلة نموذجية لاختيار الاستراتيجية (اختياري)
            mode_override:  تجاوز mode الافتراضي

        Returns:
            List[DocumentChunk]: قائمة الـ chunks النهائية مُثراة
        """
        if not text or not text.strip():
            return []

        _doc_id = doc_id or str(uuid.uuid4())
        _mode = mode_override or self.mode

        # 1. Query-aware strategy selection
        if self.config.query_aware and sample_queries:
            strategy = self._query_analyzer.analyze(text, sample_queries)
            _mode, cfg_override = self._strategy_to_mode(strategy)
        else:
            cfg_override = {}

        # 2. Choose and run chunker
        chunks = self._run_chunker(_mode, text, _doc_id, source, cfg_override)

        # 3. Metadata enrichment
        if self.config.extract_keywords or self.config.compute_chunk_scores:
            chunks = self._enricher.enrich(chunks)

        # 4. Deduplication
        if self.config.deduplication_enabled:
            chunks = self._dedup.deduplicate(chunks)

        # 5. Finalize metadata
        chunks = self._finalize(chunks, source=source)

        logger.info(
            f"[ChunkManager] doc_id={_doc_id[:8]}… | mode={_mode.value} | "
            f"chunks={len(chunks)} | source='{source}'"
        )
        return chunks

    def process_documents(
        self,
        documents: List[Document],
        *,
        sample_queries: Optional[List[str]] = None,
        mode_override: Optional[ChunkMode] = None,
    ) -> List[DocumentChunk]:
        """تقسيم قائمة مستندات"""
        all_chunks: List[DocumentChunk] = []
        for doc in documents:
            source = doc.metadata.get("source", doc.doc_id)
            chunks = self.process(
                doc.page_content,
                doc_id=doc.doc_id,
                source=source,
                sample_queries=sample_queries,
                mode_override=mode_override,
            )
            all_chunks.extend(chunks)
        return all_chunks

    def process_with_query_optimization(
        self,
        text: str,
        queries: List[str],
        doc_id: Optional[str] = None,
        source: str = "",
    ) -> List[DocumentChunk]:
        """
        Query-aware pipeline مُبسَّط.
        Simplified query-aware pipeline.
        """
        return self.process(
            text,
            doc_id=doc_id,
            source=source,
            sample_queries=queries,
        )

    # ------------------------------------------------------------------ #
    # Chunker dispatch                                                     #
    # ------------------------------------------------------------------ #

    def _run_chunker(
        self,
        mode: ChunkMode,
        text: str,
        doc_id: str,
        source: str,
        cfg_override: Dict[str, Any],
    ) -> List[DocumentChunk]:
        if mode == ChunkMode.SEMANTIC:
            return self._semantic.chunk(text, doc_id=doc_id, source=source)
        elif mode == ChunkMode.ADAPTIVE:
            return self._adaptive.chunk(text, doc_id=doc_id, source=source)
        elif mode == ChunkMode.STRUCTURAL:
            return self._structural.chunk(text, doc_id=doc_id, source=source)
        elif mode == ChunkMode.CONTEXTUAL:
            return self._contextual.chunk(text, doc_id=doc_id, source=source)
        elif mode == ChunkMode.HYBRID:
            return self._hybrid_chunk(text, doc_id, source)
        else:  # AUTO
            return self._auto_chunk(text, doc_id, source)

    def _auto_chunk(self, text: str, doc_id: str, source: str) -> List[DocumentChunk]:
        """
        اختيار تلقائي: يحلل المستند ويختار الاستراتيجية.
        Automatic selection: analyzes document and chooses strategy.
        """
        if self.config.use_semantic_chunking:
            return self._semantic.chunk(text, doc_id=doc_id, source=source)
        elif self.config.use_adaptive_chunking:
            return self._adaptive.chunk(text, doc_id=doc_id, source=source)
        elif self.config.respect_document_structure:
            return self._structural.chunk(text, doc_id=doc_id, source=source)
        elif self.config.use_context_window:
            return self._contextual.chunk(text, doc_id=doc_id, source=source)
        else:
            # Default: structural (respects paragraphs)
            return self._structural.chunk(text, doc_id=doc_id, source=source)

    def _hybrid_chunk(self, text: str, doc_id: str, source: str) -> List[DocumentChunk]:
        """
        Hybrid: structural أولاً ثم semantic على كل قسم.
        Hybrid: structural first, then semantic on each section.
        """
        structural_chunks = self._structural.chunk(text, doc_id=doc_id, source=source)
        if not structural_chunks:
            return []

        result: List[DocumentChunk] = []
        for sch in structural_chunks:
            # Run semantic within each structural chunk if large enough
            if len(sch.text) > self.config.chunk_size * 1.5:
                sub_chunks = self._semantic.chunk(sch.text, doc_id=doc_id, source=source)
                # Inherit section/heading metadata
                for sub in sub_chunks:
                    sub.metadata.section = sch.metadata.section
                    sub.metadata.heading_level = sch.metadata.heading_level
                result.extend(sub_chunks)
            else:
                result.append(sch)
        return result

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _strategy_to_mode(self, strategy: QStrategy) -> tuple:
        """تحويل QStrategy إلى ChunkMode + config overrides"""
        mode_map = {
            "faq":           ChunkMode.SEMANTIC,
            "documentation": ChunkMode.HYBRID,
            "technical":     ChunkMode.ADAPTIVE,
            "narrative":     ChunkMode.CONTEXTUAL,
            "general":       ChunkMode.STRUCTURAL,
        }
        mode = mode_map.get(strategy.name, ChunkMode.AUTO)
        override = {
            "chunk_size": strategy.chunk_size,
            "overlap":    strategy.overlap,
        }
        return mode, override

    def _finalize(self, chunks: List[DocumentChunk], source: str = "") -> List[DocumentChunk]:
        total = len(chunks)
        for i, ch in enumerate(chunks):
            ch.metadata.position = i
            ch.metadata.total_chunks = total
            if not ch.metadata.source:
                ch.metadata.source = source
        return chunks

    # ------------------------------------------------------------------ #
    # Utility                                                              #
    # ------------------------------------------------------------------ #

    def get_stats(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """إحصائيات مفيدة عن مجموعة الـ chunks"""
        if not chunks:
            return {"count": 0}
        char_counts = [c.char_count for c in chunks]
        word_counts = [c.word_count for c in chunks]
        scores = [c.metadata.score for c in chunks]
        return {
            "count":            len(chunks),
            "total_chars":      sum(char_counts),
            "avg_chars":        round(sum(char_counts) / len(chunks), 1),
            "min_chars":        min(char_counts),
            "max_chars":        max(char_counts),
            "avg_words":        round(sum(word_counts) / len(chunks), 1),
            "avg_score":        round(sum(scores) / len(chunks), 4),
            "max_score":        round(max(scores), 4),
            "chunk_types":      self._count_types(chunks),
            "unique_sections":  len({c.metadata.section for c in chunks if c.metadata.section}),
        }

    def _count_types(self, chunks: List[DocumentChunk]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for ch in chunks:
            t = ch.metadata.chunk_type.value
            counts[t] = counts.get(t, 0) + 1
        return counts
