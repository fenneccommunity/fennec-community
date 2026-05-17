# -*- coding: utf-8 -*-
"""
Directory Document Loader
"""

from typing import List, Optional, Dict, Iterator
from pathlib import Path
import fnmatch
import logging
import threading

from .base_loader import BaseDocumentLoader, BaseFileLoader, LoadedDocument
from .config_loader import DirectoryLoaderConfig, EXTENSION_MAP, LoaderType

logger = logging.getLogger(__name__)


class DirectoryLoader(BaseDocumentLoader):
    """
    Load all supported files from a directory.
    """

    def __init__(
        self,
        path: str,
        config: Optional[DirectoryLoaderConfig] = None,
        glob_pattern: str = "**/*",
        recursive: bool = True,
        silent_errors: bool = False,
        show_progress: bool = True,
    ):
        super().__init__()
        self.directory = Path(path)
        self.config = config or DirectoryLoaderConfig(
            glob_pattern=glob_pattern,
            recursive=recursive,
            silent_errors=silent_errors,
            show_progress=show_progress,
        )
        self._validate_directory()
        self._lock = threading.Lock()

    def _validate_directory(self):
        if not self.directory.exists():
            raise FileNotFoundError(f"Directory not found: {self.directory}")
        if not self.directory.is_dir():
            raise ValueError(f"Path is not a directory: {self.directory}")

    def load(self) -> List[LoadedDocument]:
        return list(self.lazy_load())

    def lazy_load(self) -> Iterator[LoadedDocument]:
        files = self._collect_files()
        total = len(files)

        if total == 0:
            logger.warning(f"No supported files found in {self.directory}")
            return

        logger.info(f"Found {total} files to load in {self.directory}")

        if self.config.use_multithreading and total > 1:
            yield from self._load_concurrent(files)
        else:
            yield from self._load_sequential(files, total)

    def _collect_files(self) -> List[Path]:
        """
        FIX: The original had broken glob logic — when recursive=True it
        stripped '**/' from the pattern and called rglob, and when
        recursive=False it stripped '**/' from the pattern and called
        glob. This meant '**/*.pdf' would never work correctly.
        
        Now: use rglob for recursive (strips leading '**/' prefix),
        and glob for non-recursive (uses basename pattern only).
                """
        pattern = self.config.glob_pattern
        supported_exts = set(EXTENSION_MAP.keys())

        if self.config.recursive:
            # rglob expects the part after '**/'
            rglob_pattern = pattern.lstrip("*/") if pattern.startswith("**/") else pattern
            all_files = list(self.directory.rglob(rglob_pattern))
        else:
            # glob with single-level pattern
            base_pattern = Path(pattern).name if "/" in pattern else pattern
            all_files = list(self.directory.glob(base_pattern))

        files = [
            f for f in all_files
            if f.is_file()
            and f.suffix.lower() in supported_exts
            and not self._is_excluded(f)
        ]
        return sorted(files)

    def _is_excluded(self, file_path: Path) -> bool:
        try:
            rel_path = str(file_path.relative_to(self.directory))
        except ValueError:
            return False
        for pattern in self.config.exclude_patterns:
            if fnmatch.fnmatch(rel_path, pattern):
                return True
            if fnmatch.fnmatch(file_path.name, pattern):
                return True
        return False

    def _load_sequential(self, files: List[Path], total: int) -> Iterator[LoadedDocument]:
        for i, file_path in enumerate(files):
            if self.config.show_progress:
                print(f"\r  Loading [{i + 1}/{total}]: {file_path.name[:50]:<50}", end="")
            yield from self._load_single_file(file_path)

        if self.config.show_progress:
            print()

    def _load_concurrent(self, files: List[Path]) -> Iterator[LoadedDocument]:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        all_docs: List[LoadedDocument] = []
        total = len(files)
        completed = 0

        with ThreadPoolExecutor(max_workers=self.config.max_concurrency) as executor:
            future_to_file = {executor.submit(self._load_single_file, f): f for f in files}

            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    docs = list(future.result())
                    with self._lock:
                        all_docs.extend(docs)
                        completed += 1
                        if self.config.show_progress:
                            print(f"\r  Loading [{completed}/{total}]: {file_path.name[:50]:<50}", end="")
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}")
                    if not self.config.silent_errors:
                        raise

        if self.config.show_progress:
            print()

        yield from all_docs

    def _load_single_file(self, file_path: Path) -> List[LoadedDocument]:
        try:
            loader = self._get_loader_for_file(file_path)
            if loader is None:
                return []
            return loader.load()
        except Exception as e:
            msg = f"Error loading {file_path}: {e}"
            if self.config.silent_errors:
                logger.warning(msg)
                return []
            raise IOError(msg) from e

    def _get_loader_for_file(self, file_path: Path) -> Optional[BaseFileLoader]:
        ext = file_path.suffix.lower()
        loader_type = EXTENSION_MAP.get(ext)
        if loader_type is None:
            return None

        fp = str(file_path)
        if loader_type == LoaderType.TEXT:
            from .text_loader import TextLoader; return TextLoader(fp)
        elif loader_type == LoaderType.MARKDOWN:
            from .text_loader import MarkdownLoader; return MarkdownLoader(fp)
        elif loader_type == LoaderType.PDF:
            from .pdf_loader import PDFLoader; return PDFLoader(fp)
        elif loader_type == LoaderType.DOCX:
            from .docx_loader import DocxLoader; return DocxLoader(fp)
        elif loader_type == LoaderType.CSV:
            from .csv_loader import CSVLoader; return CSVLoader(fp)
        elif loader_type == LoaderType.EXCEL:
            from .csv_loader import ExcelLoader; return ExcelLoader(fp)
        elif loader_type == LoaderType.JSON:
            from .json_loader import JSONLoader; return JSONLoader(fp)
        elif loader_type == LoaderType.JSONL:
            from .json_loader import JSONLinesLoader; return JSONLinesLoader(fp)
        elif loader_type == LoaderType.HTML:
            from .html_loader import HTMLLoader; return HTMLLoader(fp)
        return None

    def get_stats(self) -> Dict:
        files = self._collect_files()
        stats: Dict = {"total_files": len(files), "by_type": {}, "total_size_bytes": 0}
        for f in files:
            ext = f.suffix.lower()
            loader_type = EXTENSION_MAP.get(ext, LoaderType.AUTO)
            type_name = loader_type.value
            if type_name not in stats["by_type"]:
                stats["by_type"][type_name] = {"count": 0, "size_bytes": 0}
            stats["by_type"][type_name]["count"] += 1
            size = f.stat().st_size
            stats["by_type"][type_name]["size_bytes"] += size
            stats["total_size_bytes"] += size
        stats["total_size_mb"] = round(stats["total_size_bytes"] / (1024 * 1024), 2)
        return stats
