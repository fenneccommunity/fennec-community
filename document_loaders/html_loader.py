# -*- coding: utf-8 -*-
"""
HTML Document Loader

Supports: .html, .htm files and HTML strings

Example:
    >>> loader = HTMLLoader("page.html")
    >>> docs = loader.load()
    >>> print(docs[0].metadata["title"])
"""

from typing import List, Optional, Dict, Any
import logging
from .base_loader import BaseFileLoader, BaseDocumentLoader, LoadedDocument
from .config_loader import HTMLLoaderConfig

logger = logging.getLogger(__name__)


class HTMLLoader(BaseFileLoader):
    """
    Load HTML files and extract clean text content.\n

    Features:
        - Removes scripts, styles, nav elements 
        - Extracts title, description from meta tags 
        - Extracts links if configured 
        - Extracts table content 
        - Arabic/RTL text support 

    Example:
        >>> loader = HTMLLoader("index.html")
        >>> docs = loader.load()

        >>> # Custom config - keep tables, extract links
        >>> config = HTMLLoaderConfig(extract_links=True, extract_tables=True)
        >>> loader = HTMLLoader("page.html", config=config)
    """

    SUPPORTED_EXTENSIONS = [".html", ".htm", ".xhtml"]

    def __init__(
        self,
        file_path: str,
        config: Optional[HTMLLoaderConfig] = None,
        encoding: str = "utf-8",
    ):
        """
        Args:
            file_path: Path to HTML file 
            config: HTMLLoaderConfig 
            encoding: File encoding 
        """
        super().__init__(file_path=file_path, encoding=encoding)
        self.config = config or HTMLLoaderConfig()

    def load(self) -> List[LoadedDocument]:
        """
        Load and parse the HTML file.
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError(
                "beautifulsoup4 is required for HTMLLoader. "
                "Install it with: pip install beautifulsoup4"
            )

        # Read file | قراءة الملف
        try:
            html_content = self.file_path.read_text(
                encoding=self.encoding, errors="replace"
            )
        except Exception as e:
            raise IOError(f"Cannot read HTML file: {e}")

        return self._parse_html(html_content, source=str(self.file_path.resolve()))

    def _parse_html(self, html: str, source: str) -> List[LoadedDocument]:
        """Parse HTML string"""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, self.config.parser)

        # Extract metadata from <head> | استخراج البيانات الوصفية من <head>
        page_meta = self._extract_page_metadata(soup)

        # Remove unwanted tags | إزالة العناصر غير المرغوب فيها
        for tag in self.config.tags_to_remove:
            for element in soup.find_all(tag):
                element.decompose()

        # Extract text | استخراج النص
        if self.config.tags_to_extract:
            text_parts = []
            for tag in self.config.tags_to_extract:
                elements = soup.find_all(tag)
                for el in elements:
                    t = el.get_text(separator=" ", strip=True)
                    if t:
                        text_parts.append(t)
            text = "\n\n".join(text_parts)
        else:
            body = soup.find("body") or soup
            text = body.get_text(separator="\n", strip=True)

        # Collapse multiple blank lines | تقليص الأسطر الفارغة المتعددة
        import re
        text = re.sub(r'\n{3,}', '\n\n', text).strip()

        # Build metadata | بناء البيانات الوصفية
        meta = self._build_file_metadata(source=source)
        meta.update(page_meta)
        meta["char_count"] = len(text)

        # Extract links if configured | استخراج الروابط إذا تم التكوين
        if self.config.extract_links:
            links = [
                {"text": a.get_text(strip=True), "href": a.get("href", "")}
                for a in soup.find_all("a", href=True)
            ]
            meta["links"] = links

        documents = [LoadedDocument(page_content=text, metadata=meta)]

        # Extract tables separately if configured | استخراج الجداول منفصلة
        if self.config.extract_tables:
            table_docs = self._extract_tables(soup, source=source)
            documents.extend(table_docs)

        return documents

    def _extract_page_metadata(self, soup) -> Dict[str, Any]:
        """Extract HTML page metadata"""
        meta = {}

        # Title | العنوان
        title_tag = soup.find("title")
        if title_tag:
            meta["title"] = title_tag.get_text(strip=True)

        # Meta tags | وسوم Meta
        for tag in soup.find_all("meta"):
            name = tag.get("name", "").lower()
            prop = tag.get("property", "").lower()
            content = tag.get("content", "")

            if name in ("description", "author", "keywords"):
                meta[name] = content
            elif prop in ("og:title", "og:description", "og:url"):
                key = prop.replace("og:", "og_")
                meta[key] = content

        # Language | اللغة
        html_tag = soup.find("html")
        if html_tag and html_tag.get("lang"):
            meta["language"] = html_tag.get("lang")
        elif html_tag and html_tag.get("xml:lang"):
            meta["language"] = html_tag.get("xml:lang")

        return meta

    def _extract_tables(self, soup, source: str) -> List[LoadedDocument]:
        """Extract all tables as separate documents"""
        table_docs = []
        for i, table in enumerate(soup.find_all("table")):
            rows = []
            for tr in table.find_all("tr"):
                cells = [
                    td.get_text(strip=True)
                    for td in tr.find_all(["td", "th"])
                ]
                if any(cells):
                    rows.append(" | ".join(cells))

            if rows:
                content = "\n".join(rows)
                meta = self._build_file_metadata(
                    source=source,
                    element="table",
                    table_index=i,
                )
                table_docs.append(LoadedDocument(page_content=content, metadata=meta))

        return table_docs


class HTMLStringLoader(BaseDocumentLoader):
    """
    Load HTML from a string (not a file).\n
    Useful for processing HTML fetched from web or APIs.

    Example:
        >>> html = "<html><body><h1>Hello</h1><p>World</p></body></html>"
        >>> loader = HTMLStringLoader(html, source="https://example.com")
        >>> docs = loader.load()
    """

    def __init__(
        self,
        html_content: str,
        source: str = "html_string",
        config: Optional[HTMLLoaderConfig] = None,
    ):
        """
        Args:
            html_content: Raw HTML string 
            source: Identifier for the source 
            config: HTMLLoaderConfig 
        """
        super().__init__()
        self.html_content = html_content
        self.source = source
        self.config = config or HTMLLoaderConfig()
        # Reuse HTMLLoader's parsing logic
        self._html_loader = _HTMLParser(config=self.config)

    def load(self) -> List[LoadedDocument]:
        """Load from HTML string"""
        return self._html_loader._parse_html(self.html_content, source=self.source)


class _HTMLParser(HTMLLoader):
    """Internal helper: HTMLLoader without file validation"""
    SUPPORTED_EXTENSIONS = []

    def __init__(self, config: Optional[HTMLLoaderConfig] = None):
        # Skip file validation by calling BaseDocumentLoader directly
        BaseDocumentLoader.__init__(self)
        self.config = config or HTMLLoaderConfig()
        self.file_path = None

    def _build_file_metadata(self, source: str = "", **extra) -> Dict[str, Any]:
        meta = self._build_metadata(source=source)
        meta.update(extra)
        return meta
