"""
Full Ollama local embedding models support
This module provides integration with Ollama's local embedding models
Ollama allows you to run embedding models locally without any API keys.

Supported models  :
    - nomic-embed-text       (768 dim)  — best for Arabic + multilingual
    - mxbai-embed-large      (1024 dim) — high quality English-focused
    - all-minilm             (384 dim)  — fast & lightweight
    - snowflake-arctic-embed (1024 dim) — strong retrieval model
    - bge-m3                 (1024 dim) — multilingual, strong Arabic support
"""

from typing import List, Union, Optional, Dict, Any
import numpy as np
import logging
import time
import json
import subprocess
import sys
from .base_embedder import BaseEmbedder
from .config_embedder import EmbedderConfig
logger = logging.getLogger(__name__)
config = EmbedderConfig()


# ---------------------------------------------------------------------------
# Known model specifications | مواصفات النماذج المعروفة
# ---------------------------------------------------------------------------

MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "nomic-embed-text": {
        "dimensions": 768,
        "max_tokens": 8192,
        "language_support": "multilingual",
        "arabic_support": "good",
        "notes": "Best balance for Arabic + multilingual use",
    },
    "mxbai-embed-large": {
        "dimensions": 1024,
        "max_tokens": 512,
        "language_support": "english-primary",
        "arabic_support": "basic",
        "notes": "High quality, mainly English",
    },
    "all-minilm": {
        "dimensions": 384,
        "max_tokens": 256,
        "language_support": "multilingual",
        "arabic_support": "basic",
        "notes": "Fast and lightweight",
    },
    "snowflake-arctic-embed": {
        "dimensions": 1024,
        "max_tokens": 512,
        "language_support": "english-primary",
        "arabic_support": "basic",
        "notes": "Strong retrieval performance",
    },
    "bge-m3": {
        "dimensions": 1024,
        "max_tokens": 8192,
        "language_support": "multilingual",
        "arabic_support": "excellent",
        "notes": "Best Arabic + multilingual quality",
    },
}

DEFAULT_DIMENSIONS = 768  # fallback if model is unknown


# ---------------------------------------------------------------------------
# OllamaEmbedder
# ---------------------------------------------------------------------------

class OllamaEmbedder(BaseEmbedder):
    """
    Ollama Local Embedding Interface
    Runs embedding models entirely on your own machine — no API key needed,
    no data leaves your environment
    Features :
    - Zero-cost local inference 
    - Supports all models pulled via `ollama pull` 
    - Automatic batch processing 
    - Connection health-check 
    - Full Arabic support (model-dependent) 
    - Async-ready via BaseEmbedder 

    Requirements :
        pip install requests
        # AND Ollama must be running:  ollama serve

    Examples:
        >>> embedder = OllamaEmbedder()          # uses nomic-embed-text
        >>> emb = embedder.encode("مرحباً بك")
        >>> print(emb.shape)                     # (768,)

        >>> # Custom host / model
        >>> embedder = OllamaEmbedder(
        ...     model_name="bge-m3",
        ...     base_url="http://192.168.1.10:11434",
        ...     normalize_embeddings=True,
        ... )

        >>> # Batch encoding
        >>> texts = ["النص الأول", "النص الثاني", "النص الثالث"]
        >>> embeddings = embedder.encode(texts)
        >>> print(embeddings.shape)              # (3, 768)

        >>> # Context manager
        >>> with OllamaEmbedder("nomic-embed-text") as emb:
        ...     vecs = emb.encode(texts)
    """

    def __init__(
        self,
        model_name: str = config.embedding_model,        # "nomic-embed-text"
        base_url: str = config.base_url,                 # "http://127.0.0.1:11434"
        normalize_embeddings: bool = config.normalize_embeddings,
        batch_size: int = config.batch_size,
        max_length: Optional[int] = config.max_seq_length,
        cache_embeddings: bool = config.enable_cache,
        show_progress: bool = config.show_progress_bar,
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        auto_start_server: bool = True,
        server_start_wait: float = 3.0,
        **kwargs,
    ):
        """
        Initialize OllamaEmbedder 

        Args:
            model_name:  Ollama model name
                        e.g. "nomic-embed-text", "bge-m3", "mxbai-embed-large"
            base_url:  Ollama server URL
                      Default: "http://127.0.0.1:11434"
            normalize_embeddings:  Normalize to unit vectors
            batch_size: Batch size for processing
            max_length:  Max sequence length (info only)
            cache_embeddings:  Enable embedding cache
            show_progress: Show progress bar
            timeout:  Request timeout in seconds
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries (seconds)
            auto_start_server: Auto-start Ollama server if not running
            server_start_wait: Seconds to wait after starting the server
        """
        # Strip trailing slash from base_url
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.auto_start_server = auto_start_server
        self.server_start_wait = server_start_wait

        # Resolve embedding dimension from known specs or probe later
        self._model_spec = MODEL_SPECS.get(model_name, {})
        self._embedding_dim: Optional[int] = self._model_spec.get("dimensions")

        # Lazy-import requests before super().__init__ so _ensure_server_running can use it
        self._requests = self._import_requests()

        # ✅ Check / auto-start Ollama before anything else
        self._ensure_server_running()

        # device is "local" for Ollama — we still call super().__init__ but
        # override _get_best_device to return "local"
        super().__init__(
            model_name=model_name,
            device="local",
            normalize_embeddings=normalize_embeddings,
            batch_size=batch_size,
            max_length=max_length,
            cache_embeddings=cache_embeddings,
            show_progress=show_progress,
            **kwargs,
        )

        logger.info(f"🦙 Ollama server  : {self.base_url}")
        logger.info(f"📐 Expected dim   : {self._embedding_dim or 'auto-detect'}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _import_requests():
        """Import requests library"""
        try:
            import requests  # type: ignore
            return requests
        except ImportError as exc:
            raise ImportError(
                "❌ 'requests' is required for OllamaEmbedder.\n"
                "   Install it with:  pip install requests"
            ) from exc

    def _get_best_device(self) -> str:
        """Ollama manages its own device selection (CPU/GPU)."""
        return "local"

    def _build_url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    # ------------------------------------------------------------------
    # Server lifecycle management | إدارة دورة حياة الخادم
    # ------------------------------------------------------------------

    def _is_server_alive(self) -> bool:
        """ Quick check whether the Ollama server is reachable.
        """
        try:
            r = self._requests.get(
                self._build_url("/api/version"),
                timeout=3,
            )
            return r.status_code == 200
        except Exception:
            return False

    def _ensure_server_running(self) -> None:
        """
        Ensure Ollama server is running; auto-start it if not.

        Logic:
            1. Ping `/api/version` — if it responds, proceed.
            2. If it doesn’t respond → try starting `ollama serve` in the background.
            3. Wait for `server_start_wait` seconds and try again.
            4. If it still fails → raise a clear `ConnectionError`.

        """
        if self._is_server_alive():
            logger.info("✅ Ollama server is already running")
            return

        if not self.auto_start_server:
            raise ConnectionError(
                f"❌ Ollama server is not running at {self.base_url}.\n"
                f"   Start it manually with:  ollama serve\n"
                f"   Or pass auto_start_server=True to start automatically."
            )

        logger.warning("⚠️  Ollama server not detected — attempting to start it...")

        try:
            if sys.platform == "win32":
                # Windows: start detached process, hide console window
                proc = subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW
                    | subprocess.DETACHED_PROCESS,
                )
            else:
                # Linux / macOS: start in background
                proc = subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )

            logger.info(
                f"🚀 Started Ollama server (PID {proc.pid}) — "
                f"waiting {self.server_start_wait}s for it to be ready..."
            )

        except FileNotFoundError:
            raise ConnectionError(
                "❌ 'ollama' command not found.\n"
                "   Install Ollama from: https://ollama.com/download\n"
                "   Then re-run your code."
            )
        except Exception as e:
            raise ConnectionError(
                f"❌ Failed to start Ollama server: {e}\n"
                f"   Try starting it manually:  ollama serve"
            ) from e

        # Wait and poll until the server responds (up to server_start_wait seconds)
        deadline = time.time() + self.server_start_wait
        while time.time() < deadline:
            time.sleep(0.5)
            if self._is_server_alive():
                logger.info("✅ Ollama server started successfully")
                return

        # Final check
        if self._is_server_alive():
            logger.info("✅ Ollama server is ready")
        else:
            raise ConnectionError(
                f"❌ Ollama server did not respond within {self.server_start_wait}s "
                f"after being started.\n"
                f"   Try increasing server_start_wait or starting it manually."
            )

    # ------------------------------------------------------------------
    # Single-text API call with retry
    # ------------------------------------------------------------------

    def _embed_single(self, text: str) -> np.ndarray:
        """
        Call Ollama API for a single text with retry logic.
        """
        payload = {"model": self.model_name, "prompt": text}
        url = self._build_url("/api/embeddings")

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._requests.post(
                    url,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()

                embedding = data.get("embedding")
                if embedding is None:
                    raise ValueError(
                        f"❌ Ollama returned no 'embedding' field. "
                        f"Response keys: {list(data.keys())}"
                    )

                return np.array(embedding, dtype=np.float32)

            except self._requests.exceptions.ConnectionError as e:
                msg = (
                    f"❌ Cannot connect to Ollama at {self.base_url}.\n"
                    f"   Make sure Ollama is running:  ollama serve\n"
                    f"   Detail: {e}"
                )
                if attempt == self.max_retries:
                    raise ConnectionError(msg) from e
                logger.warning(f"⚠️  Connection error (attempt {attempt}/{self.max_retries}), retrying...")
                time.sleep(self.retry_delay * attempt)

            except self._requests.exceptions.Timeout as e:
                if attempt == self.max_retries:
                    raise TimeoutError(
                        f"❌ Ollama request timed out after {self.timeout}s"
                    ) from e
                logger.warning(f"⚠️  Timeout (attempt {attempt}/{self.max_retries}), retrying...")
                time.sleep(self.retry_delay * attempt)

            except self._requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else "?"
                if status == 404:
                    raise ValueError(
                        f"❌ Model '{self.model_name}' not found in Ollama.\n"
                        f"   Pull it first:  ollama pull {self.model_name}"
                    ) from e
                if attempt == self.max_retries:
                    raise
                logger.warning(f"⚠️  HTTP {status} (attempt {attempt}/{self.max_retries}), retrying...")
                time.sleep(self.retry_delay * attempt)

            except (ValueError, json.JSONDecodeError) as e:
                raise ValueError(f"❌ Unexpected Ollama response: {e}") from e

        # Should never reach here
        raise RuntimeError("❌ All retry attempts exhausted")  # pragma: no cover

    # ------------------------------------------------------------------
    # Core encode method (required by BaseEmbedder)
    # ------------------------------------------------------------------

    def encode(
        self,
        texts: Union[str, List[str]],
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """
        Convert texts to embeddings using Ollama.

        Args:
            texts: Single text or list of texts
            show_progress_bar:Show progress bar
            convert_to_numpy: Always numpy for Ollama
            **kwargs: Extra params (ignored)

        Returns:
            np.ndarray: shape (dim,) for single text, (n, dim) for list
        """
        start_time = time.time()
        single_text = isinstance(texts, str)

        if single_text:
            texts = [texts]

        # Optionally show progress
        use_progress = show_progress_bar or self.show_progress
        if use_progress:
            try:
                from tqdm import tqdm  # type: ignore
                iterator = tqdm(texts, desc=f"🦙 {self.model_name}", unit="text")
            except ImportError:
                iterator = texts
        else:
            iterator = texts

        all_embeddings: List[np.ndarray] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_embeddings: List[np.ndarray] = []

            for text in (iterator if use_progress else batch):
                # When iterating with tqdm we loop over all texts at once
                # so skip the inner loop re-batching
                emb = self._embed_single(text)

                # Auto-detect dimension on first successful call
                if self._embedding_dim is None:
                    self._embedding_dim = emb.shape[0]
                    logger.info(f"📐 Auto-detected embedding dim: {self._embedding_dim}")

                batch_embeddings.append(emb)

            # If tqdm consumed everything, stop outer loop
            if use_progress:
                all_embeddings = batch_embeddings
                break

            all_embeddings.extend(batch_embeddings)

        embeddings_array = np.vstack(all_embeddings).astype(np.float32)

        # Normalize if requested
        if self.normalize_embeddings:
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            embeddings_array = embeddings_array / norms

        # Update statistics
        elapsed = time.time() - start_time
        n = len(texts) if not single_text else 1
        self._stats["total_encodings"] += 1
        self._stats["total_texts"] += n
        self._stats["total_time"] += elapsed
        self._stats["avg_time_per_text"] = (
            self._stats["total_time"] / self._stats["total_texts"]
        )

        logger.debug(
            f"✅ Encoded {n} text(s) | "
            f"⏱️ {elapsed:.3f}s | "
            f"📏 dim={self.embedding_dim}"
        )

        return embeddings_array[0] if single_text else embeddings_array

    # ------------------------------------------------------------------
    # embedding_dim property (required by BaseEmbedder)
    # ------------------------------------------------------------------

    @property
    def embedding_dim(self) -> int:
        """
        Embedding vector dimension.
        If the model hasn't been called yet, performs a tiny probe request
        to determine the dimension automatically.
        """
        if self._embedding_dim is not None:
            return self._embedding_dim

        # Probe with a short test sentence
        logger.info("🔍 Probing embedding dimension from Ollama...")
        probe = self._embed_single("hello")
        self._embedding_dim = probe.shape[0]
        logger.info(f"📐 Detected dimension: {self._embedding_dim}")
        return self._embedding_dim

    # ------------------------------------------------------------------
    # Ollama-specific utilities
    # ------------------------------------------------------------------

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all models available in the running Ollama instance.

        Returns:
             List of model info dicts
        """
        url = self._build_url("/api/tags")
        try:
            response = self._requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            models = data.get("models", [])
            logger.info(f"📋 Found {len(models)} model(s) in Ollama")
            return models
        except self._requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"❌ Cannot reach Ollama at {self.base_url}. Is it running?"
            )

    def is_model_available(self, model_name: Optional[str] = None) -> bool:
        """
        Check whether a specific model is pulled and ready.

        Args:
            model_name:  Model to check (defaults to self.model_name)

        Returns:
            True True if available
        """
        target = model_name or self.model_name
        try:
            models = self.list_models()
            available_names = [m.get("name", "").split(":")[0] for m in models]
            return target in available_names or any(
                m.get("name", "").startswith(target) for m in models
            )
        except ConnectionError:
            return False

    def pull_model(self, model_name: Optional[str] = None) -> bool:
        """
        Trigger a model pull via Ollama API.

        Note:
            Ollama pull is a streaming operation. This method fires the
            request and returns True if the server accepted it. For large
            models, monitor progress with `ollama pull <name>` in a terminal.

        Args:
            model_name: Model to pull (defaults to self.model_name)

        Returns:
            True if pull was accepted
        """
        target = model_name or self.model_name
        url = self._build_url("/api/pull")
        payload = {"name": target, "stream": False}
        try:
            logger.info(f"⬇️  Pulling model '{target}' from Ollama...")
            response = self._requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            logger.info(f"✅ Model '{target}' pulled successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to pull model '{target}': {e}")
            return False

    def get_server_info(self) -> Dict[str, Any]:
        """
        Retrieve version and status information from the Ollama server.

        Returns:
            Server info dict
        """
        url = self._build_url("/api/version")
        try:
            response = self._requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except self._requests.exceptions.ConnectionError:
            return {"status": "offline", "error": "Cannot connect to Ollama"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def get_model_info(self) -> Dict[str, Any]:
        """
        Detailed model and server information.
        """
        info: Dict[str, Any] = {
            "model_name": self.model_name,
            "model_class": self.__class__.__name__,
            "embedding_dim": self.embedding_dim,
            "device": self.device,
            "base_url": self.base_url,
            "normalize_embeddings": self.normalize_embeddings,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "cache_enabled": self.cache_embeddings,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }

        if self._model_spec:
            info["model_spec"] = self._model_spec

        info["server_info"] = self.get_server_info()
        info["model_available"] = self.is_model_available()

        return info

    def validate_connection(
        self,
        test_text: str = "مرحباً Hello",
        detailed: bool = False,
    ) -> Dict[str, Any]:
        """
        Validate Ollama connection and model availability.
        Extends BaseEmbedder.validate_connection with Ollama-specific checks.
        """
        # Check server reachability first
        server_info = self.get_server_info()
        if server_info.get("status") in ("offline", "error"):
            return {
                "success": False,
                "reason": f"❌ Ollama server unreachable at {self.base_url}",
                "server_info": server_info,
                "embedding_dim": 0,
            }

        # Check model availability
        if not self.is_model_available():
            return {
                "success": False,
                "reason": (
                    f"❌ Model '{self.model_name}' not found in Ollama.\n"
                    f"   Pull it with:  ollama pull {self.model_name}"
                ),
                "server_info": server_info,
                "embedding_dim": 0,
            }

        # Run base validation (actual encode test)
        result = super().validate_connection(test_text=test_text, detailed=detailed)
        result["server_info"] = server_info
        result["base_url"] = self.base_url
        return result

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        stats = self.get_stats()
        logger.info(
            f"\n📊 OllamaEmbedder Session Summary:\n"
            f"   🔢 Total texts  : {stats['total_texts']}\n"
            f"   📝 Total calls  : {stats['total_encodings']}\n"
            f"   ⏱️  Total time   : {stats['total_time']:.2f}s\n"
            f"   📏 Avg/text     : {stats['avg_time_per_text']:.4f}s\n"
            f"   💾 Cache hits   : {stats['cache_hits']}"
        )
        self.cleanup()
        return False