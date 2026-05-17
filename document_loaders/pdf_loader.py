# -*- coding: utf-8 -*-
"""
PDF Document Loader

Supports: .pdf
Backend: PyMuPDF (fitz) → pdfplumber → pypdf (fallback chain)

Example:
    >>> loader = PDFLoader("report.pdf", per_page=True)
    >>> docs = loader.load()
    >>> print(f"Loaded {len(docs)} pages")
"""

from typing import List, Optional
import logging
from .base_loader import BaseFileLoader, LoadedDocument
from .config_loader import PDFLoaderConfig

logger = logging.getLogger(__name__)


class PDFLoader(BaseFileLoader):
    """
    Load PDF documents with multiple backend support.
    Backend Priority:
        1. PyMuPDF (fitz)   - fastest, best Arabic support
        2. pdfplumber       - best table extraction
        3. pypdf            - pure Python fallback

    Features:
        - Per-page or full-document loading 
        - Arabic/RTL text support 
        - Password-protected PDFs 
        - Page range selection 
        - Metadata extraction (title, author, etc.) 

    Example:
        >>> # Load all pages as separate documents
        >>> loader = PDFLoader("book.pdf", per_page=True)
        >>> pages = loader.load()

        >>> # Load pages 5-10 only
        >>> config = PDFLoaderConfig(start_page=4, end_page=10)
        >>> loader = PDFLoader("book.pdf", config=config)
        >>> docs = loader.load()
    """

    SUPPORTED_EXTENSIONS = [".pdf"]

    def __init__(
        self,
        file_path: str,
        config: Optional[PDFLoaderConfig] = None,
        per_page: bool = True,
        password: Optional[str] = None,
    ):
        """
        Args:
            file_path: Path to PDF file 
            config: PDFLoaderConfig instance 
            per_page: Create one Document per page 
            password: Password for encrypted PDFs 
        """
        super().__init__(file_path=file_path)
        self.config = config or PDFLoaderConfig(
            per_page=per_page,
            password=password,
        )

    def load(self) -> List[LoadedDocument]:
        """
        Load the PDF file.
        """
        # Try backends in order of preference
        backends = [
            ("fitz",       self._load_with_fitz),
            ("pdfplumber", self._load_with_pdfplumber),
            ("pypdf",      self._load_with_pypdf),
        ]

        last_error = None
        for backend_name, load_fn in backends:
            try:
                docs = load_fn()
                logger.debug(f"PDF loaded using backend: {backend_name}")
                return docs
            except ImportError:
                logger.debug(f"Backend {backend_name!r} not available")
            except Exception as e:
                logger.warning(f"Backend {backend_name!r} failed: {e}")
                last_error = e

        raise RuntimeError(
            f"All PDF backends failed for {self.file_path.name}. "
            f"Install at least one of: PyMuPDF, pdfplumber, pypdf. "
            f"Last error: {last_error}"
        )

    # ----------------------------------------------------------
    # Backend: PyMuPDF (fitz) — fastest, best Arabic support
    # ----------------------------------------------------------
    def _load_with_fitz(self) -> List[LoadedDocument]:
        """Load using PyMuPDF"""
        import fitz  # PyMuPDF

        pdf_meta = {}
        documents = []

        with fitz.open(str(self.file_path)) as pdf:
            # Unlock if password-protected | فك القفل إذا كان محمياً
            if pdf.needs_pass:
                if not self.config.password:
                    raise ValueError("PDF is password-protected. Provide 'password' in config.")
                if not pdf.authenticate(self.config.password):
                    raise ValueError("Wrong password for PDF.")

            # Extract PDF-level metadata | استخراج بيانات PDF الوصفية
            raw_meta = pdf.metadata or {}
            pdf_meta = {
                "pdf_title":   raw_meta.get("title", ""),
                "pdf_author":  raw_meta.get("author", ""),
                "pdf_subject": raw_meta.get("subject", ""),
                "total_pages": pdf.page_count,
                "backend":     "fitz",
            }

            start = self.config.start_page
            end   = self.config.end_page if self.config.end_page else pdf.page_count

            if self.config.per_page:
                for page_num in range(start, end):
                    page = pdf[page_num]
                    text = page.get_text("text").strip()
                    if not text:
                        continue
                    meta = self._build_file_metadata(
                        page_number=page_num + 1,
                        **pdf_meta,
                    )
                    documents.append(LoadedDocument(page_content=text, metadata=meta))
            else:
                sep = self.config.page_separator
                full_text = sep.join(
                    pdf[i].get_text("text").strip()
                    for i in range(start, end)
                    if pdf[i].get_text("text").strip()
                )
                meta = self._build_file_metadata(**pdf_meta)
                documents.append(LoadedDocument(page_content=full_text, metadata=meta))

        return documents

    # ----------------------------------------------------------
    # Backend: pdfplumber — good table extraction
    # ----------------------------------------------------------
    def _load_with_pdfplumber(self) -> List[LoadedDocument]:
        """Load using pdfplumber """
        import pdfplumber

        documents = []
        with pdfplumber.open(
            str(self.file_path),
            password=self.config.password or ""
        ) as pdf:
            total_pages = len(pdf.pages)
            start = self.config.start_page
            end   = self.config.end_page if self.config.end_page else total_pages

            pdf_meta = {
                "pdf_title":   pdf.metadata.get("Title", ""),
                "pdf_author":  pdf.metadata.get("Author", ""),
                "total_pages": total_pages,
                "backend":     "pdfplumber",
            }

            pages_slice = pdf.pages[start:end]

            if self.config.per_page:
                for i, page in enumerate(pages_slice):
                    text = (page.extract_text() or "").strip()
                    if not text:
                        continue
                    meta = self._build_file_metadata(
                        page_number=start + i + 1,
                        **pdf_meta,
                    )
                    documents.append(LoadedDocument(page_content=text, metadata=meta))
            else:
                sep  = self.config.page_separator
                full = sep.join(
                    (p.extract_text() or "").strip()
                    for p in pages_slice
                    if (p.extract_text() or "").strip()
                )
                meta = self._build_file_metadata(**pdf_meta)
                documents.append(LoadedDocument(page_content=full, metadata=meta))

        return documents

    # ----------------------------------------------------------
    # Backend: pypdf — pure Python fallback
    # ----------------------------------------------------------
    def _load_with_pypdf(self) -> List[LoadedDocument]:
        """Load using pypdf | التحميل باستخدام pypdf"""
        try:
            from pypdf import PdfReader
        except ImportError:
            from PyPDF2 import PdfReader  # legacy name

        reader = PdfReader(str(self.file_path))
        if reader.is_encrypted:
            if not self.config.password:
                raise ValueError("PDF is encrypted. Provide 'password' in config.")
            reader.decrypt(self.config.password)

        total_pages = len(reader.pages)
        start = self.config.start_page
        end   = self.config.end_page if self.config.end_page else total_pages

        info = reader.metadata or {}
        pdf_meta = {
            "pdf_title":   str(info.get("/Title", "")),
            "pdf_author":  str(info.get("/Author", "")),
            "total_pages": total_pages,
            "backend":     "pypdf",
        }

        documents = []
        pages = reader.pages[start:end]

        if self.config.per_page:
            for i, page in enumerate(pages):
                text = (page.extract_text() or "").strip()
                if not text:
                    continue
                meta = self._build_file_metadata(
                    page_number=start + i + 1,
                    **pdf_meta,
                )
                documents.append(LoadedDocument(page_content=text, metadata=meta))
        else:
            sep  = self.config.page_separator
            full = sep.join(
                (p.extract_text() or "").strip()
                for p in pages
                if (p.extract_text() or "").strip()
            )
            meta = self._build_file_metadata(**pdf_meta)
            documents.append(LoadedDocument(page_content=full, metadata=meta))

        return documents
