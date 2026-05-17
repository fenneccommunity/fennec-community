# -*- coding: utf-8 -*-
# ── Core Models ─────────────────────────────────────────────
from .base_loader import LoadedDocument, BaseDocumentLoader, BaseFileLoader

# ── Configuration ────────────────────────────────────────────
from .config_loader import (
    LoaderConfig,
    LoaderType,
    EXTENSION_MAP,
    TextLoaderConfig,
    PDFLoaderConfig,
    DocxLoaderConfig,
    CSVLoaderConfig,
    JSONLoaderConfig,
    HTMLLoaderConfig,
    WebLoaderConfig,
    DirectoryLoaderConfig,
)

# ── File Loaders ─────────────────────────────────────────────
from .text_loader      import TextLoader, MarkdownLoader
from .pdf_loader       import PDFLoader
from .docx_loader      import DocxLoader
from .csv_loader       import CSVLoader, ExcelLoader
from .json_loader      import JSONLoader, JSONLinesLoader
from .html_loader      import HTMLLoader, HTMLStringLoader

# ── Web Loaders ──────────────────────────────────────────────
from .web_loader       import WebLoader, MultiURLLoader

# ── Directory & Multi-Source Loaders ─────────────────────────
from .directory_loader import DirectoryLoader
from .auto_loader      import AutoLoader

# ── Public API ───────────────────────────────────────────────
__all__ = [
    # Models | النماذج
    "LoadedDocument",
    "BaseDocumentLoader",
    "BaseFileLoader",

    # Config | الإعدادات
    "LoaderConfig",
    "LoaderType",
    "EXTENSION_MAP",
    "TextLoaderConfig",
    "PDFLoaderConfig",
    "DocxLoaderConfig",
    "CSVLoaderConfig",
    "JSONLoaderConfig",
    "HTMLLoaderConfig",
    "WebLoaderConfig",
    "DirectoryLoaderConfig",

    # File Loaders | محملات الملفات
    "TextLoader",
    "MarkdownLoader",
    "PDFLoader",
    "DocxLoader",
    "CSVLoader",
    "ExcelLoader",
    "JSONLoader",
    "JSONLinesLoader",
    "HTMLLoader",
    "HTMLStringLoader",

    # Web Loaders | محملات الويب
    "WebLoader",
    "MultiURLLoader",

    # Directory & Multi | مجلد ومتعدد
    "DirectoryLoader",
    "AutoLoader",
]
