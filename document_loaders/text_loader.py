# -*- coding: utf-8 -*-
"""
Text & Markdown Document Loaders

Supports: .txt, .md, .markdown
"""

from typing import List, Optional
import logging
from .base_loader import BaseFileLoader, LoadedDocument
from .config_loader import TextLoaderConfig

logger = logging.getLogger(__name__)


class TextLoader(BaseFileLoader):
    """
    Load plain text files (.txt).    
    Features:
        - Auto-encoding detection (chardet) 
        - Handles Arabic/RTL text
        - Configurable error handling

    Example:
        >>> loader = TextLoader("document.txt")
        >>> docs = loader.load()
        >>> print(docs[0].page_content[:100])
    """

    SUPPORTED_EXTENSIONS = [".txt", ".text", ".log", ".rst"]

    def __init__(
        self,
        file_path: str,
        config: Optional[TextLoaderConfig] = None,
        encoding: str = "utf-8",
        autodetect_encoding: bool = True,
    ):
        """
        Args:
            file_path: Path to .txt file
            config: TextLoaderConfig instance 
            encoding: Preferred encoding 
            autodetect_encoding: Use chardet to detect encoding 
        """
        self.config = config or TextLoaderConfig(
            encoding=encoding,
            autodetect_encoding=autodetect_encoding,
        )
        super().__init__(file_path=file_path, encoding=self.config.encoding)

    def load(self) -> List[LoadedDocument]:
        """
        Load the text file.
        """
        text = self._read_file()
        metadata = self._build_file_metadata(
            encoding_used=self.encoding,
            char_count=len(text),
            line_count=text.count('\n') + 1,
        )
        return [LoadedDocument(page_content=text, metadata=metadata)]

    def _read_file(self) -> str:
        """Read file with fallback encoding detection"""
        # Try configured encoding first
        try:
            return self.file_path.read_text(
                encoding=self.config.encoding,
                errors=self.config.errors
            )
        except (UnicodeDecodeError, LookupError) as e:
            if not self.config.autodetect_encoding:
                raise

            # Try chardet detection
            encoding = self._detect_encoding()
            if encoding and encoding.lower() != self.config.encoding.lower():
                logger.info(
                    f"Encoding {self.config.encoding!r} failed, "
                    f"trying detected encoding {encoding!r}"
                )
                self.encoding = encoding
                return self.file_path.read_text(
                    encoding=encoding,
                    errors=self.config.errors
                )

            # Last resort: latin-1 (never fails)
            logger.warning(f"Falling back to latin-1 for {self.file_path.name}")
            self.encoding = "latin-1"
            return self.file_path.read_text(encoding="latin-1")

    def _detect_encoding(self) -> Optional[str]:
        """Detect file encoding using chardet """
        try:
            import chardet
            raw = self.file_path.read_bytes()
            result = chardet.detect(raw)
            return result.get("encoding")
        except ImportError:
            logger.debug("chardet not installed; skipping auto-detection")
            return None


class MarkdownLoader(BaseFileLoader):
    """
    Load Markdown files (.md, .markdown).

    Features:
        - Optionally strip Markdown syntax for clean text 
        - Preserve or remove code blocks 
        - Extract document title from first heading 

    Example:
        >>> loader = MarkdownLoader("README.md", strip_markdown=False)
        >>> docs = loader.load()
    """

    SUPPORTED_EXTENSIONS = [".md", ".markdown", ".mdown", ".mkd"]

    def __init__(
        self,
        file_path: str,
        encoding: str = "utf-8",
        strip_markdown: bool = False,
        remove_code_blocks: bool = False,
        autodetect_encoding: bool = True,
    ):
        """
        Args:
            file_path: Path to .md file 
            encoding: File encoding 
            strip_markdown: Remove markdown syntax 
            remove_code_blocks: Remove fenced code blocks 
            autodetect_encoding: Auto detect encoding 
        """
        super().__init__(file_path=file_path, encoding=encoding)
        self.strip_markdown = strip_markdown
        self.remove_code_blocks = remove_code_blocks
        self.autodetect_encoding = autodetect_encoding

    def load(self) -> List[LoadedDocument]:
        """
        Load the markdown file.
        """
        raw_text = self._read_file()
        title = self._extract_title(raw_text)

        # Process text
        text = raw_text
        if self.remove_code_blocks:
            text = self._remove_code_blocks(text)
        if self.strip_markdown:
            text = self._strip_markdown(text)

        metadata = self._build_file_metadata(
            title=title,
            encoding_used=self.encoding,
            char_count=len(text),
            has_code_blocks=self._has_code_blocks(raw_text),
            heading_count=raw_text.count('\n#'),
        )
        return [LoadedDocument(page_content=text, metadata=metadata)]

    def _read_file(self) -> str:
        """Read file content"""
        try:
            return self.file_path.read_text(encoding=self.encoding)
        except UnicodeDecodeError:
            if self.autodetect_encoding:
                try:
                    import chardet
                    raw = self.file_path.read_bytes()
                    detected = chardet.detect(raw).get("encoding", "latin-1")
                    self.encoding = detected
                    return self.file_path.read_text(encoding=detected)
                except ImportError:
                    pass
            return self.file_path.read_text(encoding="latin-1")

    def _extract_title(self, text: str) -> Optional[str]:
        """Extract first H1 heading as title"""
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()
        return None

    def _has_code_blocks(self, text: str) -> bool:
        """Check for fenced code blocks"""
        return "```" in text or "~~~" in text

    def _remove_code_blocks(self, text: str) -> str:
        """Remove fenced code blocks"""
        import re
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'~~~[\s\S]*?~~~', '', text)
        return text

    def _strip_markdown(self, text: str) -> str:
        """
        Strip common Markdown syntax.
        """
        import re
        # Remove headings markers
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        # Remove bold/italic
        text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
        text = re.sub(r'_{1,3}([^_]+)_{1,3}', r'\1', text)
        # Remove links [text](url) → text
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        # Remove images
        text = re.sub(r'!\[[^\]]*\]\([^)]+\)', '', text)
        # Remove inline code
        text = re.sub(r'`([^`]+)`', r'\1', text)
        # Remove horizontal rules
        text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
        # Remove blockquotes
        text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)
        # Remove list markers
        text = re.sub(r'^[\s]*[-*+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^[\s]*\d+\.\s+', '', text, flags=re.MULTILINE)
        # Collapse multiple blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
