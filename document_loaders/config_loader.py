# -*- coding: utf-8 -*-
"""
Configuration classes for Document Loaders
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class LoaderType(Enum):
    """
    Enum of all supported loader types.
    """
    TEXT      = "text"
    PDF       = "pdf"
    DOCX      = "docx"
    CSV       = "csv"
    EXCEL     = "excel"
    JSON      = "json"
    JSONL     = "jsonl"
    HTML      = "html"
    MARKDOWN  = "markdown"
    WEB       = "web"
    SITEMAP   = "sitemap"
    DIRECTORY = "directory"
    AUTO      = "auto"


# Extension → LoaderType mapping (used by AutoLoader)
# خريطة الامتداد → نوع المحمّل (تستخدمها AutoLoader)
EXTENSION_MAP: Dict[str, LoaderType] = {
    ".txt":    LoaderType.TEXT,
    ".md":     LoaderType.MARKDOWN,
    ".markdown": LoaderType.MARKDOWN,
    ".pdf":    LoaderType.PDF,
    ".docx":   LoaderType.DOCX,
    ".doc":    LoaderType.DOCX,
    ".csv":    LoaderType.CSV,
    ".tsv":    LoaderType.CSV,
    ".xlsx":   LoaderType.EXCEL,
    ".xls":    LoaderType.EXCEL,
    ".json":   LoaderType.JSON,
    ".jsonl":  LoaderType.JSONL,
    ".ndjson": LoaderType.JSONL,
    ".html":   LoaderType.HTML,
    ".htm":    LoaderType.HTML,
}


@dataclass
class TextLoaderConfig:
    """Config for TextLoader """
    encoding: str = "utf-8"
    autodetect_encoding: bool = True
    errors: str = "replace"       # "strict" | "ignore" | "replace"


@dataclass
class PDFLoaderConfig:
    """Config for PDFLoader"""
    extract_images: bool = False
    password: Optional[str] = None
    page_separator: str = "\n\n"
    per_page: bool = True          # One Document per page | مستند واحد لكل صفحة
    start_page: int = 0
    end_page: Optional[int] = None


@dataclass
class DocxLoaderConfig:
    """Config for DocxLoader"""
    include_tables: bool = True
    table_separator: str = "\n"
    row_separator: str = " | "
    include_headers: bool = True
    include_footers: bool = True


@dataclass
class CSVLoaderConfig:
    """Config for CSVLoader"""
    delimiter: str = ","
    quote_char: str = '"'
    encoding: str = "utf-8"
    source_column: Optional[str] = None      # Column to use as source | العمود المستخدم كمصدر
    content_columns: Optional[List[str]] = None  # None = all columns | None = كل الأعمدة
    metadata_columns: Optional[List[str]] = None
    skip_rows: int = 0
    max_rows: Optional[int] = None
    row_joiner: str = "\n"                   # How to join column values | كيفية ضم قيم الأعمدة


@dataclass
class JSONLoaderConfig:
    """Config for JSONLoader """
    jq_schema: Optional[str] = None          # jq path expression | تعبير مسار jq
    content_key: Optional[str] = None        # Key containing text | المفتاح الحاوي للنص
    metadata_func: Optional[Any] = None      # Callable(record) → dict
    text_content: bool = True                # If False, stringify entire JSON


@dataclass
class HTMLLoaderConfig:
    """Config for HTMLLoader"""
    parser: str = "html.parser"              # "html.parser" | "lxml" | "html5lib"
    tags_to_extract: Optional[List[str]] = None   # None = body text | None = نص الجسم
    tags_to_remove: List[str] = field(
        default_factory=lambda: ["script", "style", "nav", "footer", "header"]
    )
    extract_links: bool = False
    extract_tables: bool = True


@dataclass
class WebLoaderConfig:
    """Config for WebLoader"""
    timeout: int = 10                        # Seconds | ثواني
    headers: Dict[str, str] = field(default_factory=dict)
    verify_ssl: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0
    encoding: Optional[str] = None           # None = auto-detect | None = اكتشاف تلقائي
    extract_metadata: bool = True            # Extract OG/meta tags | استخراج وسوم Meta


@dataclass
class DirectoryLoaderConfig:
    """Config for DirectoryLoader"""
    glob_pattern: str = "**/*"
    exclude_patterns: List[str] = field(
        default_factory=lambda: ["*.pyc", "__pycache__/*", ".git/*", ".DS_Store"]
    )
    recursive: bool = True
    silent_errors: bool = False              # Skip unreadable files | تخطي الملفات غير المقروءة
    show_progress: bool = True
    max_concurrency: int = 4
    use_multithreading: bool = True


@dataclass
class LoaderConfig:
    """
    Master configuration that holds all sub-configs.
    """
    text:      TextLoaderConfig      = field(default_factory=TextLoaderConfig)
    pdf:       PDFLoaderConfig       = field(default_factory=PDFLoaderConfig)
    docx:      DocxLoaderConfig      = field(default_factory=DocxLoaderConfig)
    csv:       CSVLoaderConfig       = field(default_factory=CSVLoaderConfig)
    json:      JSONLoaderConfig      = field(default_factory=JSONLoaderConfig)
    html:      HTMLLoaderConfig      = field(default_factory=HTMLLoaderConfig)
    web:       WebLoaderConfig       = field(default_factory=WebLoaderConfig)
    directory: DirectoryLoaderConfig = field(default_factory=DirectoryLoaderConfig)
