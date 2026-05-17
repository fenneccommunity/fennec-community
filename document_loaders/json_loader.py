# -*- coding: utf-8 -*-
"""
JSON & JSONL Document Loaders

Supports: .json, .jsonl, .ndjson

Example:
    >>> loader = JSONLoader("articles.json", content_key="body")
    >>> docs = loader.load()
"""

import json
from typing import List, Optional, Dict, Any, Callable, Iterator
import logging
from .base_loader import BaseFileLoader, LoadedDocument
from .config_loader import JSONLoaderConfig

logger = logging.getLogger(__name__)


class JSONLoader(BaseFileLoader):
    """
    Load JSON files (arrays or objects).

    Handles different JSON structures:
        - Array of objects: [{...}, {...}] → each object = one Document
        - Single object: {...} → one Document
        - Nested: use jq_schema to extract data

    Features:
        - jq-style path extraction (via jq library or built-in) 
        - Custom content key selection
        - Custom metadata function
        - Arabic JSON support

    Example:
        >>> # Array of objects with "text" field
        >>> loader = JSONLoader("data.json", content_key="text")
        >>> docs = loader.load()

        >>> # With metadata function
        >>> loader = JSONLoader(
        ...     "news.json",
        ...     content_key="article_body",
        ...     metadata_func=lambda r: {"title": r.get("title"), "date": r.get("date")}
        ... )
        >>> docs = loader.load()
    """

    SUPPORTED_EXTENSIONS = [".json"]

    def __init__(
        self,
        file_path: str,
        config: Optional[JSONLoaderConfig] = None,
        content_key: Optional[str] = None,
        metadata_func: Optional[Callable[[Dict], Dict]] = None,
        jq_schema: Optional[str] = None,
        encoding: str = "utf-8",
    ):
        """
        Args:
            file_path: Path to JSON file 
            config: JSONLoaderConfig 
            content_key: Key containing main text 
            metadata_func: Function to extract metadata from record 
            jq_schema: jq path expression to filter/transform 
            encoding: File encoding 
        """
        super().__init__(file_path=file_path, encoding=encoding)
        self.config = config or JSONLoaderConfig(
            content_key=content_key,
            metadata_func=metadata_func,
            jq_schema=jq_schema,
        )

    def load(self) -> List[LoadedDocument]:
        """
        Load JSON file.
        """
        with open(self.file_path, encoding=self.encoding, errors="replace") as f:
            data = json.load(f)

        # Apply jq_schema if provided | تطبيق jq_schema إذا تم توفيره
        if self.config.jq_schema:
            data = self._apply_jq(data, self.config.jq_schema)

        # Normalize to list | تطبيع إلى قائمة
        if isinstance(data, dict):
            records = [data]
        elif isinstance(data, list):
            records = data
        else:
            # Scalar: wrap in dict
            records = [{"value": data}]

        documents = []
        for i, record in enumerate(records):
            doc = self._record_to_document(record, index=i)
            if doc:
                documents.append(doc)

        return documents

    def _record_to_document(self, record: Any, index: int) -> Optional[LoadedDocument]:
        """Convert a JSON record to a Document"""
        if isinstance(record, dict):
            # Use content_key if specified | استخدام content_key إذا تم تحديده
            if self.config.content_key and self.config.content_key in record:
                content = str(record[self.config.content_key])
            elif self.config.text_content:
                # Stringify entire record as readable text | تحويل السجل إلى نص
                content = self._dict_to_text(record)
            else:
                content = json.dumps(record, ensure_ascii=False, indent=2)
        else:
            content = str(record)

        if not content.strip():
            return None

        # Build metadata
        meta = self._build_file_metadata(record_index=index)

        if callable(self.config.metadata_func) and isinstance(record, dict):
            extra_meta = self.config.metadata_func(record)
            if isinstance(extra_meta, dict):
                meta.update(extra_meta)

        return LoadedDocument(page_content=content, metadata=meta)

    def _dict_to_text(self, d: Dict) -> str:
        """
        Convert dict to readable text (key: value format).
        """
        parts = []
        for k, v in d.items():
            if isinstance(v, (dict, list)):
                v_str = json.dumps(v, ensure_ascii=False)
            else:
                v_str = str(v) if v is not None else ""
            if v_str.strip():
                parts.append(f"{k}: {v_str}")
        return "\n".join(parts)

    def _apply_jq(self, data: Any, schema: str) -> Any:
        """
        Apply jq-style path expression.
        """
        try:
            import jq
            return jq.first(schema, data)
        except ImportError:
            # Simple dot-path fallback (e.g., ".items[].text")
            return self._simple_path(data, schema)

    def _simple_path(self, data: Any, path: str) -> Any:
        """Simple dot-notation path traversal"""
        parts = path.strip(".").split(".")
        current = data
        for part in parts:
            if not part:
                continue
            # Handle array index or key
            if isinstance(current, dict):
                current = current.get(part, {})
            elif isinstance(current, list):
                results = []
                for item in current:
                    if isinstance(item, dict):
                        val = item.get(part)
                        if val is not None:
                            results.append(val)
                current = results
        return current


class JSONLinesLoader(BaseFileLoader):
    """
    Load JSONL / NDJSON files (one JSON object per line).
    JSONL is common in datasets (Hugging Face, OpenAI fine-tuning, etc.)

    Example:
        >>> loader = JSONLinesLoader("dataset.jsonl", content_key="text")
        >>> docs = loader.load()
    """

    SUPPORTED_EXTENSIONS = [".jsonl", ".ndjson", ".jl"]

    def __init__(
        self,
        file_path: str,
        content_key: Optional[str] = None,
        metadata_func: Optional[Callable[[Dict], Dict]] = None,
        encoding: str = "utf-8",
        skip_invalid: bool = True,
        max_records: Optional[int] = None,
    ):
        """
        Args:
            file_path: Path to JSONL file 
            content_key: Key containing main text 
            metadata_func: Function to extract metadata 
            encoding: File encoding | ترميز الملف
            skip_invalid: Skip lines that fail JSON parsing 
            max_records: Maximum records to load 
        """
        super().__init__(file_path=file_path, encoding=encoding)
        self.content_key = content_key
        self.metadata_func = metadata_func
        self.skip_invalid = skip_invalid
        self.max_records = max_records

    def load(self) -> List[LoadedDocument]:
        return list(self.lazy_load())

    def lazy_load(self) -> Iterator[LoadedDocument]:
        """
        Lazily load JSONL lines.
        """
        count = 0
        with open(self.file_path, encoding=self.encoding, errors="replace") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                if self.max_records and count >= self.max_records:
                    break

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    if self.skip_invalid:
                        logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                        continue
                    raise

                # Extract content
                if self.content_key and isinstance(record, dict):
                    content = str(record.get(self.content_key, ""))
                elif isinstance(record, dict):
                    content = "\n".join(
                        f"{k}: {v}" for k, v in record.items()
                        if v and str(v).strip()
                    )
                else:
                    content = str(record)

                if not content.strip():
                    continue

                # Build metadata
                meta = self._build_file_metadata(line_number=line_num)
                if callable(self.metadata_func) and isinstance(record, dict):
                    extra = self.metadata_func(record)
                    if isinstance(extra, dict):
                        meta.update(extra)

                yield LoadedDocument(page_content=content, metadata=meta)
                count += 1

        logger.debug(f"Loaded {count} documents from {self.file_path.name}")
