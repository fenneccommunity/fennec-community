"""
Respects document structure (headers, paragraphs, code blocks, lists) during chunking
"""
from __future__ import annotations
import uuid
import logging
from typing import List, Optional

from .base import BaseChunker
from .doc_model import DocumentChunk, ChunkMetadata, ChunkType, DocumentType
from .chunk_config import ChunkConfig
from .document_structure_parser import DocumentStructureParser, StructuredSection

logger = logging.getLogger(__name__)


class StructureAwareChunker(BaseChunker):
    """


    Sections that exceed max_chunk_size are split further by sentences.
    Short sections are merged with the next one (unless they're headings).

    Splits the document while respecting its natural structure:
    - Markdown: headers → sections, code blocks preserved
    - HTML: H1-H6 → sections, lists/tables preserved
    - Plain text: paragraphs
    """

    def __init__(
        self,
        config: Optional[ChunkConfig] = None,
        *,
        max_chunk_size: int = 1024,
        min_chunk_size: int = 50,
        language: str = "auto",
        split_on_headers: bool = True,
        preserve_code_blocks: bool = True,
    ) -> None:
        cfg = config or ChunkConfig()
        self.max_chunk_size = max_chunk_size if config is None else cfg.max_chunk_size
        self.min_chunk_size = min_chunk_size if config is None else cfg.min_chunk_size
        self.language = language
        self.split_on_headers = split_on_headers if config is None else cfg.split_on_headers
        self.preserve_code_blocks = preserve_code_blocks if config is None else cfg.preserve_code_blocks
        self._parser = DocumentStructureParser()

    # ------------------------------------------------------------------ #
    # BaseChunker implementation                                           #
    # ------------------------------------------------------------------ #

    def _chunk_impl(self, text: str, doc_id: str, source: str) -> List[DocumentChunk]:
        if not text or not text.strip():
            return []

        doc_type = self._parser.detect_document_type(text)
        sections = self._parser.parse(text, doc_type)

        if not sections:
            return []

        chunks: List[DocumentChunk] = []
        position = 0
        current_heading = ""
        current_level = 0

        # Merge/split sections into proper chunks
        merged_sections = self._merge_small_sections(sections)

        for sec in merged_sections:
            if sec.heading_level > 0:
                current_heading = sec.heading
                current_level = sec.heading_level

            # Code blocks: keep as single chunk regardless of size
            if sec.is_code_block and self.preserve_code_blocks:
                chunks.append(self._make_chunk(
                    sec.text, doc_id=doc_id, source=source,
                    position=position,
                    heading=current_heading,
                    heading_level=current_level,
                    chunk_type=ChunkType.CODE_BLOCK,
                    doc_type=doc_type,
                    is_header=sec.heading_level > 0 and sec.extra.get("is_heading_node", False),
                    heading_level_val=sec.heading_level,
                    extra=sec.extra,
                ))
                position += 1
                continue

            # Tables / lists: keep as single chunk
            if sec.is_table or sec.is_list:
                chunks.append(self._make_chunk(
                    sec.text, doc_id=doc_id, source=source,
                    position=position,
                    heading=current_heading,
                    heading_level=current_level,
                    chunk_type=ChunkType.TABLE if sec.is_table else ChunkType.LIST_ITEM,
                    doc_type=doc_type,
                    extra=sec.extra,
                ))
                position += 1
                continue

            # Section is a heading node itself
            if sec.extra.get("is_heading_node", False):
                chunks.append(self._make_chunk(
                    sec.text, doc_id=doc_id, source=source,
                    position=position,
                    heading=sec.heading,
                    heading_level=sec.heading_level,
                    chunk_type=ChunkType.HEADER,
                    doc_type=doc_type,
                    is_header=True,
                    heading_level_val=sec.heading_level,
                    extra=sec.extra,
                ))
                position += 1
                continue

            # Normal text section — split if too large
            sub_texts = self._split_large_section(sec.text)
            for sub in sub_texts:
                if sub.strip():
                    chunks.append(self._make_chunk(
                        sub, doc_id=doc_id, source=source,
                        position=position,
                        heading=current_heading,
                        heading_level=current_level,
                        chunk_type=ChunkType.SECTION,
                        doc_type=doc_type,
                        extra=sec.extra,
                    ))
                    position += 1

        return chunks

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _merge_small_sections(self, sections: List[StructuredSection]) -> List[StructuredSection]:
        """دمج الأقسام القصيرة مع التالي (إلا إذا كان header أو code)"""
        if not sections:
            return []
        result: List[StructuredSection] = []
        buffer_text = ""
        buffer_sec: Optional[StructuredSection] = None

        def flush():
            nonlocal buffer_text, buffer_sec
            if buffer_sec and buffer_text.strip():
                buffer_sec.text = buffer_text.strip()
                result.append(buffer_sec)
            buffer_text = ""
            buffer_sec = None

        for sec in sections:
            is_special = sec.is_code_block or sec.is_table or sec.is_list or sec.extra.get("is_heading_node")

            if is_special:
                flush()
                result.append(sec)
                continue

            candidate = (buffer_text + " " + sec.text).strip()
            if len(candidate) < self.min_chunk_size:
                buffer_text = candidate
                if buffer_sec is None:
                    buffer_sec = sec
            elif len(buffer_text) < self.min_chunk_size and buffer_sec is not None:
                buffer_text = candidate
            else:
                flush()
                buffer_text = sec.text
                buffer_sec = sec

        flush()
        return result

    def _split_large_section(self, text: str) -> List[str]:
        """تقسيم قسم كبير إلى sub-chunks حسب الجمل"""
        if len(text) <= self.max_chunk_size:
            return [text]

        import re
        sentences = re.split(r'(?<=[.!?؟\u06D4])\s+|\n{2,}', text)
        result: List[str] = []
        current = ""
        for sent in sentences:
            if len(current) + len(sent) > self.max_chunk_size and current:
                result.append(current.strip())
                current = sent
            else:
                current = (current + " " + sent).strip()
        if current:
            result.append(current.strip())
        return result

    def _make_chunk(
        self,
        text: str,
        doc_id: str,
        source: str,
        position: int,
        heading: str = "",
        heading_level: int = 0,
        chunk_type: ChunkType = ChunkType.SECTION,
        doc_type: DocumentType = DocumentType.PLAIN_TEXT,
        is_header: bool = False,
        heading_level_val: int = 0,
        extra: dict = None,
    ) -> DocumentChunk:
        return DocumentChunk(
            text=text,
            chunk_id=str(uuid.uuid4()),
            doc_id=doc_id,
            metadata=ChunkMetadata(
                source=source,
                section=heading,
                position=position,
                chunk_type=chunk_type,
                document_type=doc_type,
                language=self.language,
                is_header=is_header,
                heading_level=heading_level_val,
                char_start=0,
                char_end=len(text),
                extra=extra or {},
            ),
        )
