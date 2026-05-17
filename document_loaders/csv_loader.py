# -*- coding: utf-8 -*-
"""
CSV & Excel Document Loaders  (improved)
"""

import csv
from typing import List, Optional, Dict, Iterator
import logging
from .base_loader import BaseFileLoader, LoadedDocument
from .config_loader import CSVLoaderConfig

logger = logging.getLogger(__name__)


class CSVLoader(BaseFileLoader):
    """
    Load CSV/TSV files with flexible column mapping.
    """

    SUPPORTED_EXTENSIONS = [".csv", ".tsv"]

    def __init__(
        self,
        file_path: str,
        config: Optional[CSVLoaderConfig] = None,
        content_columns: Optional[List[str]] = None,
        metadata_columns: Optional[List[str]] = None,
        source_column: Optional[str] = None,
        encoding: str = "utf-8",
        delimiter: str = ",",
    ):
        super().__init__(file_path=file_path, encoding=encoding)
        self.config = config or CSVLoaderConfig(
            encoding=encoding,
            delimiter=delimiter,
            content_columns=content_columns,
            metadata_columns=metadata_columns,
            source_column=source_column,
        )
        if self.file_path.suffix.lower() == ".tsv" and delimiter == ",":
            self.config.delimiter = "\t"

    def load(self) -> List[LoadedDocument]:
        return list(self.lazy_load())

    def lazy_load(self) -> Iterator[LoadedDocument]:
        """
        FIX: The original caught UnicodeDecodeError and called lazy_load()
        recursively after mutating self.config.encoding. This could silently
        loop or yield duplicate documents if the retry itself also failed.
        Now we detect encoding BEFORE opening, using a single retry with
        latin-1 as a guaranteed fallback — no recursion.
        """
        encodings_to_try = [self.config.encoding, "latin-1"]
        last_error = None

        for enc in encodings_to_try:
            try:
                yield from self._iter_rows(enc)
                return
            except UnicodeDecodeError as e:
                last_error = e
                logger.warning(f"Encoding '{enc}' failed for {self.file_path.name}, trying next...")

        raise UnicodeDecodeError(
            last_error.encoding, last_error.object, last_error.start,
            last_error.end, f"All encodings failed for {self.file_path.name}"
        ) from last_error

    def _iter_rows(self, encoding: str) -> Iterator[LoadedDocument]:
        """Internal: iterate rows with a specific encoding"""
        with open(self.file_path, newline="", encoding=encoding, errors="strict") as f:
            for _ in range(self.config.skip_rows):
                next(f, None)

            reader = csv.DictReader(
                f,
                delimiter=self.config.delimiter,
                quotechar=self.config.quote_char,
            )

            row_count = 0
            for i, row in enumerate(reader):
                if self.config.max_rows and i >= self.config.max_rows:
                    break
                doc = self._row_to_document(row, row_number=i + 1)
                if doc:
                    yield doc
                    row_count += 1

        logger.debug(f"Loaded {row_count} documents from {self.file_path.name}")

    def _row_to_document(self, row: Dict[str, str], row_number: int) -> Optional[LoadedDocument]:
        if self.config.content_columns:
            content_parts = [
                f"{col}: {row.get(col, '').strip()}"
                for col in self.config.content_columns
                if row.get(col, "").strip()
            ]
            content = self.config.row_joiner.join(content_parts)
        else:
            content = self.config.row_joiner.join(
                f"{k}: {v}" for k, v in row.items() if v and v.strip()
            )

        if not content.strip():
            return None

        meta = self._build_file_metadata(row_number=row_number)

        if self.config.metadata_columns:
            for col in self.config.metadata_columns:
                if col in row:
                    meta[col] = row[col]
        elif self.config.content_columns:
            for k, v in row.items():
                if k not in self.config.content_columns:
                    meta[k] = v

        if self.config.source_column and self.config.source_column in row:
            meta["source"] = row[self.config.source_column]

        return LoadedDocument(page_content=content, metadata=meta)


class ExcelLoader(BaseFileLoader):
    """
    Load Excel files (.xlsx, .xls).
    """

    SUPPORTED_EXTENSIONS = [".xlsx", ".xls", ".xlsm"]

    def __init__(
        self,
        file_path: str,
        sheet_name: Optional[str] = None,
        load_all_sheets: bool = False,
        content_columns: Optional[List[str]] = None,
        metadata_columns: Optional[List[str]] = None,
        skip_rows: int = 0,
        max_rows: Optional[int] = None,
    ):
        super().__init__(file_path=file_path)
        self.sheet_name = sheet_name
        self.load_all_sheets = load_all_sheets
        self.content_columns = content_columns
        self.metadata_columns = metadata_columns
        self.skip_rows = skip_rows
        self.max_rows = max_rows

    def load(self) -> List[LoadedDocument]:
        """
        Added .xls support via xlrd engine, and .xlsm via openpyxl.
        """
        ext = self.file_path.suffix.lower()

        # Legacy .xls requires xlrd engine
        if ext == ".xls":
            return self._load_with_pandas()

        try:
            import openpyxl
        except ImportError:
            raise ImportError("openpyxl is required. Install: pip install openpyxl")

        wb = openpyxl.load_workbook(str(self.file_path), read_only=True, data_only=True)
        documents = []

        if self.load_all_sheets:
            sheet_names = wb.sheetnames
        elif self.sheet_name:
            if self.sheet_name not in wb.sheetnames:
                raise ValueError(
                    f"Sheet '{self.sheet_name}' not found. "
                    f"Available: {wb.sheetnames}"
                )
            sheet_names = [self.sheet_name]
        else:
            sheet_names = [wb.sheetnames[0]]

        for sname in sheet_names:
            ws = wb[sname]
            documents.extend(self._load_sheet(ws, sname))

        wb.close()
        return documents

    def _load_with_pandas(self) -> List[LoadedDocument]:
        """Fallback for .xls files using xlrd | دعم ملفات .xls"""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for .xls files. Install: pip install pandas xlrd")

        xls = pd.ExcelFile(str(self.file_path), engine="xlrd")
        sheet_names = xls.sheet_names if self.load_all_sheets else [self.sheet_name or xls.sheet_names[0]]
        documents = []
        for sname in sheet_names:
            df = xls.parse(sname, skiprows=self.skip_rows, nrows=self.max_rows)
            for i, row in df.iterrows():
                row_dict = {str(k): str(v) for k, v in row.items() if str(v) != "nan"}
                parts = [
                    f"{k}: {v}" for k, v in row_dict.items()
                    if (not self.content_columns or k in self.content_columns) and v.strip()
                ]
                content = "\n".join(parts)
                if not content.strip():
                    continue
                meta = self._build_file_metadata(sheet_name=sname, row_number=i + 2)
                if self.metadata_columns:
                    for col in self.metadata_columns:
                        meta[col] = row_dict.get(col, "")
                documents.append(LoadedDocument(page_content=content, metadata=meta))
        return documents

    def _load_sheet(self, ws, sheet_name: str) -> List[LoadedDocument]:
        documents = []
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            return []

        headers = [
            str(h).strip() if h is not None else f"col_{i}"
            for i, h in enumerate(rows[0])
        ]

        data_rows = rows[1 + self.skip_rows:]
        if self.max_rows:
            data_rows = data_rows[:self.max_rows]

        for row_idx, row in enumerate(data_rows):
            row_dict = {
                headers[i]: str(val).strip() if val is not None else ""
                for i, val in enumerate(row)
                if i < len(headers)
            }

            if self.content_columns:
                parts = [
                    f"{col}: {row_dict.get(col, '')}"
                    for col in self.content_columns
                    if row_dict.get(col, "").strip()
                ]
            else:
                parts = [f"{k}: {v}" for k, v in row_dict.items() if v.strip()]

            content = "\n".join(parts)
            if not content.strip():
                continue

            meta = self._build_file_metadata(sheet_name=sheet_name, row_number=row_idx + 2)
            if self.metadata_columns:
                for col in self.metadata_columns:
                    meta[col] = row_dict.get(col, "")

            documents.append(LoadedDocument(page_content=content, metadata=meta))

        return documents
