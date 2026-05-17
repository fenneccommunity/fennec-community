"""
Parses document structure and extracts structured sections for HTML, Markdown, and plain text
"""
from __future__ import annotations
import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from .doc_model import DocumentType

logger = logging.getLogger(__name__)


@dataclass
class StructuredSection:
    """structured section of a document, with metadata for structure-aware chunking"""
    text: str
    heading: str = ""
    heading_level: int = 0          # 0 = body, 1-6 = H1-H6
    page: Optional[int] = None
    section_index: int = 0
    is_code_block: bool = False
    is_list: bool = False
    is_table: bool = False
    char_start: int = 0
    char_end: int = 0
    extra: dict = field(default_factory=dict)


class DocumentStructureParser:
    """
    Parses Markdown, HTML, and plain-text documents, returning a list of
    StructuredSection objects that downstream chunkers use to make structure-aware splits.
    """

    # ------------------------------------------------------------------ #
    # Public entry point                                                   #
    # ------------------------------------------------------------------ #

    def parse(self, text: str, doc_type: DocumentType) -> List[StructuredSection]:
        """
        Parse the document and return a list of sections.
        """
        if doc_type == DocumentType.MARKDOWN:
            return self._parse_markdown(text)
        elif doc_type == DocumentType.HTML:
            return self._parse_html(text)
        else:
            return self._parse_plain(text)

    def detect_document_type(self, text: str) -> DocumentType:
        """
        Heuristically detect document type from content.
        """
        # Check for HTML
        if re.search(r'<(html|body|div|p|h[1-6]|ul|ol|li|table)[^>]*>', text, re.IGNORECASE):
            return DocumentType.HTML
        # Check for Markdown headers / code blocks / bold
        if re.search(r'^#{1,6}\s', text, re.MULTILINE) or \
           re.search(r'```', text) or \
           re.search(r'\*\*.+?\*\*', text):
            return DocumentType.MARKDOWN
        # Check for Arabic
        if re.search(r'[\u0600-\u06FF]', text):
            arabic_ratio = len(re.findall(r'[\u0600-\u06FF]', text)) / max(len(text), 1)
            if arabic_ratio > 0.2:
                return DocumentType.ARABIC
        return DocumentType.PLAIN_TEXT

    # ------------------------------------------------------------------ #
    # Markdown parser                                                      #
    # ------------------------------------------------------------------ #

    def _parse_markdown(self, text: str) -> List[StructuredSection]:
        sections: List[StructuredSection] = []
        lines = text.split("\n")
        current_heading = ""
        current_level = 0
        current_lines: List[str] = []
        in_code_block = False
        code_fence_re = re.compile(r'^```')
        heading_re = re.compile(r'^(#{1,6})\s+(.*)')
        char_pos = 0
        section_index = 0

        def flush(extra: dict = None):
            nonlocal section_index
            body = "\n".join(current_lines).strip()
            if body:
                sec = StructuredSection(
                    text=body,
                    heading=current_heading,
                    heading_level=current_level,
                    section_index=section_index,
                    is_code_block=False,
                    extra=extra or {},
                )
                sections.append(sec)
                section_index += 1
            current_lines.clear()

        for line in lines:
            # Toggle code block
            if code_fence_re.match(line):
                if in_code_block:
                    current_lines.append(line)
                    body = "\n".join(current_lines).strip()
                    if body:
                        sections.append(StructuredSection(
                            text=body,
                            heading=current_heading,
                            heading_level=current_level,
                            section_index=section_index,
                            is_code_block=True,
                        ))
                        section_index += 1
                    current_lines.clear()
                    in_code_block = False
                else:
                    flush()
                    in_code_block = True
                    current_lines.append(line)
                continue

            if in_code_block:
                current_lines.append(line)
                continue

            m = heading_re.match(line)
            if m:
                flush()
                current_heading = m.group(2).strip()
                current_level = len(m.group(1))
                # The heading itself is a small section
                sections.append(StructuredSection(
                    text=current_heading,
                    heading=current_heading,
                    heading_level=current_level,
                    section_index=section_index,
                    is_code_block=False,
                    extra={"is_heading_node": True},
                ))
                section_index += 1
            else:
                current_lines.append(line)

        flush()
        return [s for s in sections if s.text.strip()]

    # ------------------------------------------------------------------ #
    # HTML parser                                                          #
    # ------------------------------------------------------------------ #

    def _parse_html(self, text: str) -> List[StructuredSection]:
        try:
            from bs4 import BeautifulSoup
            return self._parse_html_bs4(text)
        except ImportError:
            logger.warning("[DocumentStructureParser] BeautifulSoup4 not installed. Using regex HTML parser.")
            return self._parse_html_regex(text)

    def _parse_html_bs4(self, text: str) -> List[StructuredSection]:
        from bs4 import BeautifulSoup, NavigableString, Tag

        soup = BeautifulSoup(text, "html.parser")
        sections: List[StructuredSection] = []
        section_index = 0
        current_heading = ""
        current_level = 0

        heading_tags = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}

        for element in soup.find_all(True):
            tag_name = element.name.lower() if element.name else ""

            if tag_name in heading_tags:
                heading_text = element.get_text(strip=True)
                current_heading = heading_text
                current_level = heading_tags[tag_name]
                sections.append(StructuredSection(
                    text=heading_text,
                    heading=heading_text,
                    heading_level=current_level,
                    section_index=section_index,
                    extra={"tag": tag_name, "is_heading_node": True},
                ))
                section_index += 1

            elif tag_name in ("p", "blockquote"):
                body = element.get_text(separator=" ", strip=True)
                if body:
                    sections.append(StructuredSection(
                        text=body,
                        heading=current_heading,
                        heading_level=current_level,
                        section_index=section_index,
                        extra={"tag": tag_name},
                    ))
                    section_index += 1

            elif tag_name in ("ul", "ol"):
                items = [li.get_text(separator=" ", strip=True) for li in element.find_all("li")]
                body = "\n".join(f"• {it}" for it in items if it)
                if body:
                    sections.append(StructuredSection(
                        text=body,
                        heading=current_heading,
                        heading_level=current_level,
                        section_index=section_index,
                        is_list=True,
                        extra={"tag": tag_name},
                    ))
                    section_index += 1

            elif tag_name == "pre":
                code = element.get_text()
                if code.strip():
                    sections.append(StructuredSection(
                        text=code,
                        heading=current_heading,
                        heading_level=current_level,
                        section_index=section_index,
                        is_code_block=True,
                        extra={"tag": "pre"},
                    ))
                    section_index += 1

            elif tag_name == "table":
                rows = []
                for tr in element.find_all("tr"):
                    cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                    rows.append(" | ".join(cells))
                body = "\n".join(rows)
                if body:
                    sections.append(StructuredSection(
                        text=body,
                        heading=current_heading,
                        heading_level=current_level,
                        section_index=section_index,
                        is_table=True,
                        extra={"tag": "table"},
                    ))
                    section_index += 1

        return [s for s in sections if s.text.strip()]

    def _parse_html_regex(self, text: str) -> List[StructuredSection]:
        """Fallback: strip tags and split on double newlines"""
        clean = re.sub(r'<[^>]+>', ' ', text)
        clean = re.sub(r'\s+', ' ', clean).strip()
        return self._parse_plain(clean)

    # ------------------------------------------------------------------ #
    # Plain text parser                                                    #
    # ------------------------------------------------------------------ #

    def _parse_plain(self, text: str) -> List[StructuredSection]:
        """تقسيم نص عادي على أساس الفقرات (سطران فارغان أو أكثر)"""
        paragraphs = re.split(r'\n{2,}', text)
        sections: List[StructuredSection] = []
        for i, para in enumerate(paragraphs):
            para = para.strip()
            if para:
                sections.append(StructuredSection(
                    text=para,
                    section_index=i,
                ))
        return sections
