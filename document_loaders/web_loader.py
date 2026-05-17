# -*- coding: utf-8 -*-
"""
Web URL Document Loader  (improved)
"""

from typing import List, Optional, Dict
import logging
import time
import threading
from .base_loader import BaseDocumentLoader, LoadedDocument
from .config_loader import WebLoaderConfig
from .html_loader import HTMLStringLoader, HTMLLoaderConfig

logger = logging.getLogger(__name__)


class WebLoader(BaseDocumentLoader):
    """
    Load content from web URLs.
    """

    def __init__(
        self,
        url: str,
        config: Optional[WebLoaderConfig] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 10,
    ):
        super().__init__()
        self.url = url.strip()
        self.config = config or WebLoaderConfig(headers=headers or {}, timeout=timeout)

    def load(self) -> List[LoadedDocument]:
        html = self._fetch_url()
        html_config = HTMLLoaderConfig(
            tags_to_remove=["script", "style", "nav", "footer", "header", "aside"],
            extract_links=False,
            extract_tables=True,
        )
        loader = HTMLStringLoader(html, source=self.url, config=html_config)
        docs = loader.load()
        for doc in docs:
            doc.metadata["url"] = self.url
            doc.metadata["source"] = self.url
        return docs

    def _fetch_url(self) -> str:
        try:
            import requests
        except ImportError:
            raise ImportError("requests is required. Install: pip install requests")

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; DocumentLoader/1.0; "
                "+https://github.com/your-lib)"
            ),
            **self.config.headers,
        }

        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                response = requests.get(
                    self.url,
                    headers=headers,
                    timeout=self.config.timeout,
                    verify=self.config.verify_ssl,
                )
                response.raise_for_status()
                response.encoding = self.config.encoding or response.apparent_encoding or "utf-8"
                return response.text

            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    wait = self.config.retry_delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed for {self.url}: {e}. Retrying in {wait:.1f}s...")
                    time.sleep(wait)

        raise ConnectionError(
            f"Failed to fetch {self.url} after {self.config.max_retries} attempts. Last error: {last_error}"
        )


class MultiURLLoader(BaseDocumentLoader):
    """
    Load content from multiple URLs concurrently.
    """

    def __init__(
        self,
        urls: List[str],
        config: Optional[WebLoaderConfig] = None,
        max_workers: int = 4,
        delay_between_requests: float = 0.5,
        continue_on_error: bool = True,
    ):
        super().__init__()
        self.urls = [url.strip() for url in urls if url.strip()]
        self.config = config or WebLoaderConfig()
        self.max_workers = max_workers
        self.delay = delay_between_requests
        self.continue_on_error = continue_on_error

    def load(self) -> List[LoadedDocument]:
        """
          load content from multiple URLs concurrently.
    
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        all_docs: List[LoadedDocument] = []
        errors = []

        # Semaphore + last_request_time enforce global rate limiting
        sem = threading.Semaphore(1)
        last_time = {"t": 0.0}

        def _load_with_throttle(url: str) -> List[LoadedDocument]:
            with sem:
                elapsed = time.monotonic() - last_time["t"]
                if elapsed < self.delay:
                    time.sleep(self.delay - elapsed)
                last_time["t"] = time.monotonic()
            # Fetch outside the semaphore so downloads run in parallel
            return WebLoader(url=url, config=self.config).load()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_url = {executor.submit(_load_with_throttle, url): url for url in self.urls}

            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    docs = future.result()
                    all_docs.extend(docs)
                    logger.info(f"✓ Loaded: {url} ({len(docs)} docs)")
                except Exception as e:
                    errors.append((url, str(e)))
                    logger.error(f"✗ Failed: {url} - {e}")
                    if not self.continue_on_error:
                        raise

        if errors:
            logger.warning(
                f"Failed to load {len(errors)}/{len(self.urls)} URLs. "
                f"Failed: {[u for u, _ in errors]}"
            )

        logger.info(
            f"MultiURLLoader: loaded {len(all_docs)} documents "
            f"from {len(self.urls) - len(errors)}/{len(self.urls)} URLs"
        )
        return all_docs
