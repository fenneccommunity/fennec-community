"""
Enhanced Arabic text chunker, now integrated with the AI-Powered Chunking System.
Preserves all original capabilities + adds SemanticChunker/AdaptiveChunker integration.
"""
from __future__ import annotations
import re
import inspect
import warnings
import logging
import uuid
from typing import List, Optional, Tuple, Any

from .base import BaseChunker, ChunkingStrategy
from .doc_model import DocumentChunk, ChunkMetadata, ChunkType, DocumentType
from .chunk_config import ChunkConfig

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

config = ChunkConfig()


# ────────────────────────────────────────────────────────────────────────
# Lazy torch/transformers loader
# ────────────────────────────────────────────────────────────────────────

def _load_torch_and_transformers():
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
        return torch, AutoTokenizer, AutoModel
    except ImportError as e:
        raise ImportError(
            "⚠ Required libraries not found: torch and transformers.\n"
            "pip install torch transformers\n"
            f"Original error: {e}"
        ) from e


# ────────────────────────────────────────────────────────────────────────
# ArabicTextChunker
# ────────────────────────────────────────────────────────────────────────

class ArabicTextChunker(BaseChunker, ChunkingStrategy):
    """

    Smart Arabic text chunker supporting:
    - Rule-based splitting
    - Semantic chunking (BERT-based)
    - Smart overlap
    - Strict size limits
    - Automatic spacing fix and normalization
    - Full Arabic and mixed-language text support
    """

    SENTENCE_DELIMITERS = r'[.!?؟\n]'
    ARABIC_PATTERN = re.compile(r'[\u0600-\u06FF]+')

    SPACE_PATTERNS = {
        'after_punct':      (r'([.!?؟،,;:])([^\s])',          r'\1 \2'),
        'before_paren':     (r'([^\s])(\()',                   r'\1 \2'),
        'after_paren':      (r'(\))([^\s.,!?؟])',              r'\1 \2'),
        'word_boundary':    (r'([\u0600-\u06FF]+)([A-Za-z]+)', r'\1 \2'),
        'reverse_boundary': (r'([A-Za-z]+)([\u0600-\u06FF]+)', r'\1 \2'),
    }

    def __init__(
        self,
        chunk_size: int = config.chunk_size,
        overlap: int = config.overlap,
        min_chunk_size: int = config.min_chunk_size,
        model_name: str = config.model_name,
        use_semantic_chunking: bool = config.use_semantic_chunking,
        device: Optional[str] = None,
        preserve_formatting: bool = config.preserve_formatting,
        fix_spacing: bool = config.fix_spacing,
        strict_size_limit: bool = config.strict_size_limit,
        size_tolerance: float = config.size_tolerance,
        use_smart_overlap: bool = config.use_smart_overlap,
        smart_overlap_threshold: float = config.smart_overlap_threshold,
    ) -> None:
        self._validate_parameters(chunk_size, overlap, min_chunk_size)

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self.use_semantic_chunking = use_semantic_chunking
        self.preserve_formatting = preserve_formatting
        self.fix_spacing = fix_spacing
        self.strict_size_limit = strict_size_limit
        self.size_tolerance = size_tolerance
        self.use_smart_overlap = use_smart_overlap
        self.smart_overlap_threshold = smart_overlap_threshold

        torch, AutoTokenizer, AutoModel = _load_torch_and_transformers()
        self._torch = torch
        self._AutoTokenizer = AutoTokenizer
        self._AutoModel = AutoModel

        self.device = device or ("cuda" if self._torch.cuda.is_available() else "cpu")
        self.effective_max_size = int(self.chunk_size * (1 + self.size_tolerance))

        self._initialize_model(model_name)

    # ------------------------------------------------------------------ #
    # Class helpers                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_parameters(chunk_size: int, overlap: int, min_chunk_size: int) -> None:
        if chunk_size <= 0 or overlap < 0:
            raise ValueError("chunk_size must be positive and overlap non-negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")
        if min_chunk_size <= 0 or min_chunk_size > chunk_size:
            raise ValueError("Invalid min_chunk_size")

    @classmethod
    def get_available_parameters(cls) -> List[str]:
        sig = inspect.signature(cls.__init__)
        params = list(sig.parameters.keys())
        params.remove("self")
        return params

    @classmethod
    def create_safely(cls, **kwargs) -> "ArabicTextChunker":
        available = cls.get_available_parameters()
        filtered = {k: v for k, v in kwargs.items() if k in available}
        rejected = set(kwargs) - set(filtered)
        if rejected:
            logger.warning(f"[ArabicTextChunker] Unsupported parameters ignored: {rejected}")
        return cls(**filtered)

    # ------------------------------------------------------------------ #
    # Model initialization                                                 #
    # ------------------------------------------------------------------ #

    def _initialize_model(self, model_name: str) -> None:
        try:
            self.tokenizer = self._AutoTokenizer.from_pretrained(model_name)
            if self.use_semantic_chunking:
                self.model = self._AutoModel.from_pretrained(model_name)
                self.model.to(self.device)
                self.model.eval()
            else:
                self.model = None
            logger.info(f"[ArabicTextChunker] ✓ Model loaded: {model_name} on {self.device}")
        except Exception as exc:
            logger.warning(f"[ArabicTextChunker] ⚠ Failed to load model: {exc}. Using rule-based mode.")
            self.tokenizer = None
            self.model = None
            self.use_semantic_chunking = False
            self.use_smart_overlap = False

    # ------------------------------------------------------------------ #
    # BaseChunker / ChunkingStrategy implementation                       #
    # ------------------------------------------------------------------ #

    def chunk(self, text: str, doc_id: str = "", source: str = "") -> List[DocumentChunk]:
        """Unified entry point (satisfies both BaseChunker and ChunkingStrategy)"""
        raw = self._chunk_impl(text, doc_id=doc_id, source=source)
        return self._finalize(raw, total=len(raw))

    def _chunk_impl(self, text: str, doc_id: str, source: str) -> List[DocumentChunk]:
        if not text or not text.strip():
            return []

        text = self._preprocess(text)

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
                    document_type=DocumentType.ARABIC,
                    language="arabic",
                    char_start=char_cursor,
                    char_end=char_cursor + len(chunk_text),
                ),
            ))
            char_cursor += len(chunk_text) + 1
        return chunks

    # ------------------------------------------------------------------ #
    # Text preprocessing                                                   #
    # ------------------------------------------------------------------ #

    def _preprocess(self, text: str) -> str:
        if self.fix_spacing:
            text = self._fix_text_spacing(text)
        text = self._normalize_text(text)
        return text

    def _fix_text_spacing(self, text: str) -> str:
        if not self.fix_spacing:
            return text
        for _, (pattern, replacement) in self.SPACE_PATTERNS.items():
            text = re.sub(pattern, replacement, text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s+([.,!?؟،;:])', r'\1', text)
        return text.strip()

    def _normalize_text(self, text: str) -> str:
        """تطبيع النص العربي"""
        # Normalize Alef variants
        text = re.sub(r'[إأآا]', 'ا', text)
        # Normalize Teh Marbuta
        text = re.sub(r'ة', 'ه', text)
        # Normalize Yeh variants
        text = re.sub(r'[يى]', 'ي', text)
        # Remove diacritics (tashkeel) — optional but helps similarity
        text = re.sub(r'[\u064B-\u065F\u0610-\u061A\u06D6-\u06DC]', '', text)
        # Clean multiple spaces/newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]{2,}', ' ', text)
        return text.strip()

    # ------------------------------------------------------------------ #
    # Rule-based chunking                                                  #
    # ------------------------------------------------------------------ #

    def _rule_based_chunk(self, text: str) -> List[str]:
        sentences = self._split_into_sentences(text)
        if not sentences:
            return [text] if text.strip() else []

        chunks: List[str] = []
        current_sentences: List[str] = []
        current_len = 0

        for sentence in sentences:
            sent_len = len(sentence)

            if current_len + sent_len > self.effective_max_size and current_sentences:
                chunk_text = " ".join(current_sentences)
                if self.strict_size_limit and len(chunk_text) > self.effective_max_size:
                    sub_chunks = self._hard_split(chunk_text)
                    chunks.extend(sub_chunks)
                else:
                    chunks.append(chunk_text)

                # Compute smart / standard overlap
                overlap_sentences = self._compute_overlap(current_sentences, chunks)
                current_sentences = overlap_sentences
                current_len = sum(len(s) for s in current_sentences)

            current_sentences.append(sentence)
            current_len += sent_len

        if current_sentences:
            chunk_text = " ".join(current_sentences)
            if self.strict_size_limit and len(chunk_text) > self.effective_max_size:
                chunks.extend(self._hard_split(chunk_text))
            else:
                chunks.append(chunk_text)

        return [c for c in chunks if c.strip()]

    def _split_into_sentences(self, text: str) -> List[str]:
        """تقسيم النص إلى جمل مع الحفاظ على التنسيق"""
        parts = re.split(self.SENTENCE_DELIMITERS, text)
        sentences = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            # If a part is too long, split on commas/semicolons
            if len(part) > self.chunk_size:
                sub_parts = re.split(r'[،,;؛]', part)
                sentences.extend([s.strip() for s in sub_parts if s.strip()])
            else:
                sentences.append(part)
        return sentences

    def _compute_overlap(
        self,
        current_sentences: List[str],
        all_chunks: List[str],
    ) -> List[str]:
        """احسب الجمل للـ overlap (ذكي أو عادي)"""
        if self.use_smart_overlap and self.model is not None and all_chunks:
            return self._smart_overlap_sentences(current_sentences, all_chunks[-1])
        # Standard: keep last N chars worth of sentences
        overlap_buf: List[str] = []
        overlap_len = 0
        for s in reversed(current_sentences):
            if overlap_len + len(s) <= self.overlap:
                overlap_buf.insert(0, s)
                overlap_len += len(s)
            else:
                break
        return overlap_buf

    def _hard_split(self, text: str) -> List[str]:
        """تقسيم قسري على حدود الكلمات"""
        chunks: List[str] = []
        words = text.split()
        current: List[str] = []
        current_len = 0
        for word in words:
            if current_len + len(word) + 1 > self.chunk_size and current:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
            current.append(word)
            current_len += len(word) + 1
        if current:
            chunks.append(" ".join(current))
        return chunks

    # ------------------------------------------------------------------ #
    # Semantic chunking (BERT-based)                                       #
    # ------------------------------------------------------------------ #

    def _semantic_chunk(self, text: str) -> List[str]:
        """تقسيم دلالي باستخدام BERT + cosine similarity"""
        sentences = self._split_into_sentences(text)
        if len(sentences) < 2:
            return self._rule_based_chunk(text)

        try:
            embeddings = self._get_embeddings(sentences)
            chunks: List[str] = []
            current_sentences: List[str] = [sentences[0]]
            current_len = len(sentences[0])

            for i in range(1, len(sentences)):
                similarity = self._cosine_similarity(embeddings[i - 1], embeddings[i])

                if (
                    similarity < self.smart_overlap_threshold
                    or current_len + len(sentences[i]) > self.effective_max_size
                ) and current_sentences:
                    chunks.append(" ".join(current_sentences))
                    overlap_sents = self._smart_overlap_sentences(current_sentences, chunks[-1])
                    current_sentences = overlap_sents
                    current_len = sum(len(s) for s in current_sentences)

                current_sentences.append(sentences[i])
                current_len += len(sentences[i])

            if current_sentences:
                chunks.append(" ".join(current_sentences))

            return [c for c in chunks if c.strip()] or self._rule_based_chunk(text)

        except Exception as exc:
            logger.warning(f"[ArabicTextChunker] Semantic chunking failed: {exc}. Using rule-based.")
            return self._rule_based_chunk(text)

    def _get_embeddings(self, sentences: List[str]) -> List[Any]:
        """استخراج BERT embeddings للجمل"""
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
            # Mean pooling
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

    def _smart_overlap_sentences(
        self,
        sentences: List[str],
        previous_chunk: str,
    ) -> List[str]:
        """اختيار جمل الـ overlap بناءً على التشابه الدلالي"""
        if not self.model or not sentences:
            return []
        try:
            prev_emb = self._get_embeddings([previous_chunk])[0]
            scored: List[Tuple[float, str]] = []
            for sent in sentences[-min(5, len(sentences)):]:
                sent_emb = self._get_embeddings([sent])[0]
                sim = self._cosine_similarity(prev_emb, sent_emb)
                scored.append((sim, sent))
            # Keep sentences above threshold
            selected = [s for score, s in sorted(scored, key=lambda x: -x[0])
                        if score >= self.smart_overlap_threshold]
            # Respect overlap size limit
            result: List[str] = []
            total = 0
            for s in selected:
                if total + len(s) <= self.overlap:
                    result.append(s)
                    total += len(s)
                else:
                    break
            return result
        except Exception:
            return []

    # ------------------------------------------------------------------ #
    # Utility                                                              #
    # ------------------------------------------------------------------ #

    def get_arabic_ratio(self, text: str) -> float:
        """arabic_chars / total_chars ratio"""
        arabic_chars = len(self.ARABIC_PATTERN.findall(text))
        return arabic_chars / max(len(text), 1)

    def is_arabic_dominant(self, text: str, threshold: float = 0.3) -> bool:
        return self.get_arabic_ratio(text) >= threshold
