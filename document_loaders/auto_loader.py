# -*- coding: utf-8 -*-
"""
Auto Document Loader - Smart Loader that detects file type automatically
المحمّل التلقائي - يكتشف نوع الملف تلقائياً

The AutoLoader is the recommended entry point for most use cases.
AutoLoader هو نقطة الدخول الموصى بها لمعظم حالات الاستخدام.

Example:
    >>> from document_loaders import AutoLoader
    >>> 
    >>> # Automatically picks the right loader
    >>> docs = AutoLoader.load("report.pdf")
    >>> docs = AutoLoader.load("data.csv")
    >>> docs = AutoLoader.load("https://example.com/page")
    >>> docs = AutoLoader.load("./my_documents/")
"""

from typing import List
from pathlib import Path
import logging

from .base_loader import BaseDocumentLoader,  LoadedDocument
from .config_loader import EXTENSION_MAP, LoaderType

logger = logging.getLogger(__name__)


class AutoLoader:
    """
    Smart loader that automatically selects the correct loader.\n
    المحمّل الذكي الذي يختار تلقائياً المحمّل الصحيح.

    Supports all file types and URLs without needing to know which
    loader to use. Just pass a path, URL, or directory.

    Supported inputs:
        📄 Files:      .txt, .md, .pdf, .docx, .csv, .xlsx, .json, .jsonl, .html
        🌐 URLs:       http://, https://
        📁 Directories: any directory path

    Example:
        >>> # Files
        >>> docs = AutoLoader.load("report.pdf")
        >>> docs = AutoLoader.load("article.md")
        >>> docs = AutoLoader.load("dataset.csv")
        >>> docs = AutoLoader.load("page.html")

        >>> # URL
        >>> docs = AutoLoader.load("https://docs.python.org")

        >>> # Directory
        >>> docs = AutoLoader.load("./my_docs/")

        >>> # With options
        >>> docs = AutoLoader.load(
        ...     "big_report.pdf",
        ...     per_page=True,
        ...     password="secret"
        ... )
    """

    @classmethod
    def load(cls, source: str, **kwargs) -> List[LoadedDocument]:
        """
        Load documents from any source (file, URL, or directory).\n
        تحميل المستندات من أي مصدر (ملف، URL، أو مجلد).

        Args:
            source: File path, URL, or directory path | مسار ملف، URL، أو مجلد
            **kwargs: Additional arguments passed to the specific loader |
                      وسيطات إضافية تُمرَّر للمحمّل المحدد

        Returns:
            List[LoadedDocument]: Loaded documents | المستندات المحملة

        Raises:
            ValueError: If source type cannot be determined | إذا تعذر تحديد نوع المصدر
            FileNotFoundError: If file/directory doesn't exist | إذا لم يوجد الملف/المجلد
        """
        loader = cls.get_loader(source, **kwargs)
        return loader.load()

    @classmethod
    def get_loader(cls, source: str, **kwargs) -> BaseDocumentLoader:
        """
        Get the appropriate loader for a source without loading.\n
        الحصول على المحمّل المناسب للمصدر دون تحميل.

        Useful when you want to configure the loader before loading.

        Args:
            source: File path, URL, or directory | مسار ملف، URL، أو مجلد

        Returns:
            BaseDocumentLoader: The appropriate loader | المحمّل المناسب
        """
        source = str(source).strip()

        # 1. URL Detection | اكتشاف URL
        if source.startswith(("http://", "https://", "ftp://")):
            return cls._create_web_loader(source, **kwargs)

        path = Path(source)

        # 2. Directory Detection | اكتشاف المجلد
        if path.is_dir():
            return cls._create_directory_loader(source, **kwargs)

        # 3. File Detection by Extension | اكتشاف الملف بالامتداد
        return cls._create_file_loader(path, **kwargs)

    @classmethod
    def _create_web_loader(cls, url: str, **kwargs) -> BaseDocumentLoader:
        """Create web loader | إنشاء محمّل ويب"""
        from .web_loader import WebLoader
        return WebLoader(url=url, **kwargs)

    @classmethod
    def _create_directory_loader(cls, path: str, **kwargs) -> BaseDocumentLoader:
        """Create directory loader | إنشاء محمّل مجلدات"""
        from .directory_loader import DirectoryLoader
        return DirectoryLoader(path=path, **kwargs)

    @classmethod
    def _create_file_loader(cls, path: Path, **kwargs) -> BaseDocumentLoader:
        """Create file loader based on extension | إنشاء محمّل ملفات بناءً على الامتداد"""
        ext = path.suffix.lower()
        loader_type = EXTENSION_MAP.get(ext)

        if loader_type is None:
            raise ValueError(
                f"Unsupported file type: '{ext}' for file {path.name}\n"
                f"Supported extensions: {sorted(EXTENSION_MAP.keys())}"
            )

        file_path = str(path)

        if loader_type == LoaderType.TEXT:
            from .text_loader import TextLoader
            return TextLoader(file_path, **kwargs)

        elif loader_type == LoaderType.MARKDOWN:
            from .text_loader import MarkdownLoader
            return MarkdownLoader(file_path, **kwargs)

        elif loader_type == LoaderType.PDF:
            from .pdf_loader import PDFLoader
            return PDFLoader(file_path, **kwargs)

        elif loader_type == LoaderType.DOCX:
            from .docx_loader import DocxLoader
            return DocxLoader(file_path, **kwargs)

        elif loader_type == LoaderType.CSV:
            from .csv_loader import CSVLoader
            return CSVLoader(file_path, **kwargs)

        elif loader_type == LoaderType.EXCEL:
            from .csv_loader import ExcelLoader
            return ExcelLoader(file_path, **kwargs)

        elif loader_type == LoaderType.JSON:
            from .json_loader import JSONLoader
            return JSONLoader(file_path, **kwargs)

        elif loader_type == LoaderType.JSONL:
            from .json_loader import JSONLinesLoader
            return JSONLinesLoader(file_path, **kwargs)

        elif loader_type in (LoaderType.HTML,):
            from .html_loader import HTMLLoader
            return HTMLLoader(file_path, **kwargs)

        raise ValueError(f"No loader implemented for type: {loader_type}")

    @classmethod
    def detect_type(cls, source: str) -> str:
        """
        Detect the type of a source without loading.\n
        اكتشاف نوع المصدر دون تحميل.

        Returns:
            str: Type name (e.g., "pdf", "csv", "web", "directory") | اسم النوع
        """
        source = str(source).strip()

        if source.startswith(("http://", "https://", "ftp://")):
            return "web"

        path = Path(source)
        if path.is_dir():
            return "directory"

        ext = path.suffix.lower()
        loader_type = EXTENSION_MAP.get(ext)
        if loader_type:
            return loader_type.value

        return "unknown"


