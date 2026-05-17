"""
Enhanced multilingual text chunker, integrated with the AI-Powered Chunking System.
Preserves all original capabilities + smart overlap + strict size limits.
"""
from __future__ import annotations
import re
import uuid
import warnings
import logging
from typing import List, Optional, Dict, Any, Tuple

from .base import BaseChunker
from .doc_model import DocumentChunk, ChunkMetadata, ChunkType, DocumentType
from .chunk_config import ChunkConfig

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

config = ChunkConfig()


# ────────────────────────────────────────────────────────────────────────
# Lazy loader
# ────────────────────────────────────────────────────────────────────────

def _load_torch_and_transformers():
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModel
        return torch, AutoTokenizer, AutoModelForTokenClassification, AutoModel
    except ImportError as e:
        raise ImportError(
            "⚠ Required libraries not found: torch and transformers.\n"
            "pip install torch transformers\n"
            f"Original error: {e}"
        ) from e


class MultilanguageTextChunker(BaseChunker):
    """
    Multilingual text chunker supporting:
    - Auto language detection
    - Semantic chunking (BERT-based)
    - Smart overlap
    - Strict size limits
    - 17+ languages
    """

    SENTENCE_DELIMITERS = r'[.!?。！？।।॥\n]'

    LANGUAGE_PATTERNS: Dict[str, re.Pattern] = {
        'arabic':    re.compile(r'[\u0600-\u06FF]+'),
        'chinese':   re.compile(r'[\u4e00-\u9fff]+'),
        'japanese':  re.compile(r'[\u3040-\u309f\u30a0-\u30ff]+'),
        'korean':    re.compile(r'[\uac00-\ud7af]+'),
        'cyrillic':  re.compile(r'[\u0400-\u04FF]+'),
        'devanagari':re.compile(r'[\u0900-\u097F]+'),
        'thai':      re.compile(r'[\u0E00-\u0E7F]+'),
        'hebrew':    re.compile(r'[\u0590-\u05FF]+'),
    }

    LANGUAGE_MODELS: Dict[str, str] = {
        'multilingual': config.multilingual,
        'english':      config.english,
        'arabic':       config.arabic,
        'chinese':      config.chinese,
        'french':       config.french,
        'german':       config.german,
        'spanish':      config.spanish,
        'russian':      config.russian,
        'japanese':     config.japanese,
        'korean':       config.korean,
        'portuguese':   config.portuguese,
        'italian':      config.italian,
        'dutch':        config.dutch,
        'polish':       config.polish,
        'turkish':      config.turkish,
        'vietnamese':   config.vietnamese,
        'hindi':        config.hindi,
    }

    def __init__(
        self,
        chunk_size: int = config.chunk_size,
        overlap: int = config.overlap,
        min_chunk_size: int = config.min_chunk_size,
        model_name: Optional[str] = None,
        language: str = 'auto',
        use_semantic_chunking: bool = config.use_semantic_chunking,
        device: Optional[str] = None,
        use_smart_overlap: bool = True,
        smart_overlap_threshold: float = 0.7,
        strict_size_limit: bool = True,
        size_tolerance: float = 0.05,
    ) -> None:
        if chunk_size <= 0 or overlap < 0:
            raise ValueError("chunk_size must be positive and overlap non-negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")
        if min_chunk_size <= 0 or min_chunk_size > chunk_size:
            raise ValueError("Invalid min_chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self.use_semantic_chunking = use_semantic_chunking
        self.language = language.lower()
        self.use_smart_overlap = use_smart_overlap
        self.smart_overlap_threshold = smart_overlap_threshold
        self.strict_size_limit = strict_size_limit
        self.size_tolerance = size_tolerance
        self.effective_max_size = int(chunk_size * (1 + size_tolerance))

        torch, AutoTokenizer, AutoModelForTokenClassification, AutoModel = _load_torch_and_transformers()
        self._torch = torch
        self._AutoTokenizer = AutoTokenizer
        self._AutoModelForTokenClassification = AutoModelForTokenClassification
        self._AutoModel = AutoModel

        self.device = device or ("cuda" if self._torch.cuda.is_available() else "cpu")

        if model_name is None:
            model_name = (
                self.LANGUAGE_MODELS.get(self.language, self.LANGUAGE_MODELS['multilingual'])
                if self.language != 'auto'
                else self.LANGUAGE_MODELS['multilingual']
            )
        self.model_name = model_name
        self._initialize_model()

    # ------------------------------------------------------------------ #
    # Model initialization                                                 #
    # ------------------------------------------------------------------ #

    def _initialize_model(self) -> None:
        try:
            self.tokenizer = self._AutoTokenizer.from_pretrained(self.model_name)
            if self.use_semantic_chunking:
                self.model = self._AutoModel.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model.eval()
            else:
                self.model = None
            logger.info(f"[MultilanguageTextChunker] ✓ Loaded '{self.model_name}' on {self.device}")
        except Exception as exc:
            logger.warning(f"[MultilanguageTextChunker] ⚠ Failed to load model: {exc}. Using rule-based mode.")
            self.tokenizer = None
            self.model = None
            self.use_semantic_chunking = False
            self.use_smart_overlap = False

    # ------------------------------------------------------------------ #
    # BaseChunker implementation                                           #
    # ------------------------------------------------------------------ #

    def _chunk_impl(self, text: str, doc_id: str, source: str) -> List[DocumentChunk]:
        if not text or not text.strip():
            return []

        detected_lang = self._detect_language(text) if self.language == 'auto' else self.language

        if self.use_semantic_chunking and self.model is not None:
            raw_texts = self._semantic_chunk(text)
        else:
            raw_texts = self._rule_based_chunk(text)

        chunks: List[DocumentChunk] = []
        char_cursor = 0
        for i, chunk_text in enumerate(raw_texts):
            chunk_text = chunk_text.strip()
            if len(chunk_text) < self.min_chunk_size:
                char_cursor += len(chunk_text) + 1
                continue
            chunks.append(DocumentChunk(
                text=chunk_text,
                chunk_id=str(uuid.uuid4()),
                doc_id=doc_id,
                metadata=ChunkMetadata(
                    source=source,
                    position=i,
                    chunk_type=ChunkType.SEMANTIC if self.use_semantic_chunking else ChunkType.PARAGRAPH,
                    document_type=DocumentType.PLAIN_TEXT,
                    language=detected_lang,
                    char_start=char_cursor,
                    char_end=char_cursor + len(chunk_text),
                ),
            ))
            char_cursor += len(chunk_text) + 1
        return chunks

    # ------------------------------------------------------------------ #
    # Language detection                                                   #
    # ------------------------------------------------------------------ #

    def _detect_language(self, text: str) -> str:
        if not text:
            return 'english'
        counts: Dict[str, int] = {}
        for lang, pattern in self.LANGUAGE_PATTERNS.items():
            matches = pattern.findall(text)
            counts[lang] = sum(len(m) for m in matches)

        if not counts:
            return 'english'
        dominant = max(counts, key=lambda k: counts[k])
        if counts[dominant] == 0:
            return 'english'

        lang_map = {
            'cyrillic': 'russian',
            'devanagari': 'hindi',
        }
        return lang_map.get(dominant, dominant)

    # ------------------------------------------------------------------ #
    # Rule-based chunking                                                  #
    # ------------------------------------------------------------------ #

    def _rule_based_chunk(self, text: str) -> List[str]:
        sentences = [s.strip() for s in re.split(self.SENTENCE_DELIMITERS, text) if s.strip()]
        if not sentences:
            return [text] if text.strip() else []

        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        for sent in sentences:
            sent_len = len(sent)
            if current_len + sent_len > self.effective_max_size and current:
                chunk_text = ' '.join(current)
                if self.strict_size_limit and len(chunk_text) > self.effective_max_size:
                    chunks.extend(self._hard_split(chunk_text))
                else:
                    chunks.append(chunk_text)

                # Overlap
                overlap_buf: List[str] = []
                overlap_len = 0
                for s in reversed(current):
                    if overlap_len + len(s) <= self.overlap:
                        overlap_buf.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break
                current = overlap_buf
                current_len = overlap_len

            current.append(sent)
            current_len += sent_len

        if current:
            chunk_text = ' '.join(current)
            if self.strict_size_limit and len(chunk_text) > self.effective_max_size:
                chunks.extend(self._hard_split(chunk_text))
            else:
                chunks.append(chunk_text)

        return [c for c in chunks if c.strip()]

    def _hard_split(self, text: str) -> List[str]:
        chunks: List[str] = []
        words = text.split()
        current: List[str] = []
        current_len = 0
        for word in words:
            if current_len + len(word) + 1 > self.chunk_size and current:
                chunks.append(' '.join(current))
                current = []
                current_len = 0
            current.append(word)
            current_len += len(word) + 1
        if current:
            chunks.append(' '.join(current))
        return chunks

    # ------------------------------------------------------------------ #
    # Semantic chunking                                                    #
    # ------------------------------------------------------------------ #

    def _semantic_chunk(self, text: str) -> List[str]:
        sentences = [s.strip() for s in re.split(self.SENTENCE_DELIMITERS, text) if s.strip()]
        if len(sentences) < 2:
            return self._rule_based_chunk(text)
        try:
            embeddings = self._get_embeddings(sentences)
            chunks: List[str] = []
            current: List[str] = [sentences[0]]
            current_len = len(sentences[0])

            for i in range(1, len(sentences)):
                sim = self._cosine_similarity(embeddings[i - 1], embeddings[i])
                if (sim < self.smart_overlap_threshold or
                        current_len + len(sentences[i]) > self.effective_max_size) and current:
                    chunk_text = ' '.join(current)
                    if self.strict_size_limit and len(chunk_text) > self.effective_max_size:
                        chunks.extend(self._hard_split(chunk_text))
                    else:
                        chunks.append(chunk_text)

                    # Overlap
                    overlap_buf: List[str] = []
                    overlap_len = 0
                    for s in reversed(current):
                        if overlap_len + len(s) <= self.overlap:
                            overlap_buf.insert(0, s)
                            overlap_len += len(s)
                        else:
                            break
                    current = overlap_buf
                    current_len = overlap_len

                current.append(sentences[i])
                current_len += len(sentences[i])

            if current:
                chunk_text = ' '.join(current)
                if self.strict_size_limit and len(chunk_text) > self.effective_max_size:
                    chunks.extend(self._hard_split(chunk_text))
                else:
                    chunks.append(chunk_text)

            return [c for c in chunks if c.strip()] or self._rule_based_chunk(text)

        except Exception as exc:
            logger.warning(f"[MultilanguageTextChunker] Semantic failed: {exc}. Fallback to rule-based.")
            return self._rule_based_chunk(text)

    def _get_embeddings(self, sentences: List[str]) -> List[Any]:
        embeddings = []
        for sent in sentences:
            inputs = self.tokenizer(
                sent,
                return_tensors=config.return_tensor,
                max_length=config.max_len,
                padding=config.padding,
                truncation=config.truncation,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with self._torch.no_grad():
                outputs = self.model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            embeddings.append(emb)
        return embeddings

    def _cosine_similarity(self, a: Any, b: Any) -> float:
        import numpy as np
        a, b = np.array(a), np.array(b)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    # ------------------------------------------------------------------ #
    # Utility                                                              #
    # ------------------------------------------------------------------ #

    def get_supported_languages(self) -> List[str]:
        return list(self.LANGUAGE_MODELS.keys())

    def switch_language(self, language: str) -> None:
        """switch to a different language model (if supported)"""
        self.language = language.lower()
        new_model = self.LANGUAGE_MODELS.get(self.language, self.LANGUAGE_MODELS['multilingual'])
        if new_model != self.model_name:
            self.model_name = new_model
            self._initialize_model()
