"""
AI-Powered Intelligent Chunking System
Production-Grade RAG Chunking Library
"""

# ── Core models ────────────────────────────────────────────────────────
from .doc_model import (
    DocumentChunk,
    Document,
    ChunkMetadata,
    ChunkType,
    DocumentType,
)

# ── Configuration ──────────────────────────────────────────────────────
from .chunk_config import ChunkConfig

# ── Abstract bases ─────────────────────────────────────────────────────
from .base import BaseChunker, TextSplitter, ChunkingStrategy

# ── Chunkers ───────────────────────────────────────────────────────────
from .semantic_chunker import SemanticChunker
from .adaptive_chunker import AdaptiveChunker
from .structure_aware_chunker import StructureAwareChunker
from .context_aware_chunker import ContextAwareChunker
from .arabic_chunker import ArabicTextChunker
from .multi_chunker import MultilanguageTextChunker

# ── Orchestrator ───────────────────────────────────────────────────────
from .chunk_manager import ChunkManager, ChunkMode

# ── Text splitters ─────────────────────────────────────────────────────
from .text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    SentenceTextSplitter,
)

# ── Supporting components ──────────────────────────────────────────────
from .embeddings import EmbeddingProvider , cosine_similarity_matrix
from .metadata_enricher import MetadataEnricher
from .deduplicator import Deduplicator
from .document_structure_parser import DocumentStructureParser, StructuredSection
from .query_pattern_analyzer import QueryPatternAnalyzer

__all__ = [
    # Models
    "DocumentChunk",
    "Document",
    "ChunkMetadata",
    "ChunkType",
    "DocumentType",

    # Config
    "ChunkConfig",

    # Bases
    "BaseChunker",
    "TextSplitter",
    "ChunkingStrategy",

    # Chunkers
    "SemanticChunker",
    "AdaptiveChunker",
    "StructureAwareChunker",
    "ContextAwareChunker",
    "ArabicTextChunker",
    "MultilanguageTextChunker",

    # Orchestrator
    "ChunkManager",
    "ChunkMode",

    # Text splitters
    "CharacterTextSplitter",
    "RecursiveCharacterTextSplitter",
    "TokenTextSplitter",
    "SentenceTextSplitter",

    # Components
    "EmbeddingProvider",
    "cosine_similarity_matrix",
    "MetadataEnricher",
    "Deduplicator",
    "DocumentStructureParser",
    "StructuredSection",
    "QueryPatternAnalyzer",
]

