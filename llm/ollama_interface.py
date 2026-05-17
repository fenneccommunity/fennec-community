"""
Ollama LLM Interface
"""
import asyncio
import subprocess
import sys
import requests
import time
from typing import Optional
from .base_llm_interface import BaseLLMInterface
from .config_llm import llm_config
import logging
logger = logging.getLogger(__name__)
config = llm_config()


class OllamaInterface(BaseLLMInterface):

    def __init__(self, model_name: str = config.ollama_model, api_key: Optional[str] = None, **kwargs):
        super().__init__(model_name, api_key, **kwargs)
        self.base_url = kwargs.get("base_url", config.ollama_base_url)
        self.auto_start = kwargs.get("auto_start", True)
        self.server_start_wait = kwargs.get("server_start_wait", 10)

        # ✅ Track whether WE started the server (so we only stop what we started)
        self._we_started_server = False
        self.server_process = None

        if self._check_server_status():
            print(f"✓ Connected to Ollama server at {self.base_url}")
        elif self.auto_start:
            print("⚠️ Ollama server not running, attempting to start...")
            self._start_server()
        else:
            print("⚠️ Ollama server not running. Start manually with: ollama serve")

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    def _check_server_status(self) -> bool:
        """ Quick server health check."""
        try:
            response = requests.get(f"{self.base_url}/api/version", timeout=3)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def _start_server(self) -> bool:
        """
        Start Ollama in the background with smart polling.
        """
        try:
            if sys.platform == "win32":
                self.server_process = subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS,
                )
            else:
                self.server_process = subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )

            self._we_started_server = True
            logger.info(f"⏳ Waiting for Ollama server to start (PID {self.server_process.pid})...")

            # Smart polling: check every 0.5s instead of sleeping a full second
            deadline = time.time() + self.server_start_wait
            attempt = 0
            while time.time() < deadline:
                time.sleep(0.5)
                attempt += 1
                if self._check_server_status():
                    logger.info(f"✓ Ollama server started successfully! (after {attempt * 0.5:.1f}s)")
                    return True

            logger.warning(f"❌ Ollama server did not respond within {self.server_start_wait}s")
            return False

        except FileNotFoundError:
            logger.warning("❌ Ollama not found. Install from: https://ollama.com/download")
            return False
        except Exception as e:
            logger.warning(f"❌ Error starting Ollama server: {e}")
            return False

    def stop_server(self):
        """
        Stop server only if we started it — never kill a pre-existing instance.
        """
        if self.server_process and self._we_started_server:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
                logger.info("✓ Ollama server stopped")
            except Exception as e:
                logger.warning(f"⚠️ Error stopping server: {e}")
                self.server_process.kill()
            finally:
                self.server_process = None
                self._we_started_server = False

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, prompt: str, max_tokens: int = config.max_token,
                 temperature: float = config.temperature, **kwargs) -> str:
        """
        Generate text using streaming internally to prevent timeout on slow/large models.
        Each token resets the read-timeout clock, so long responses never time out mid-generation.
        """
        try:
            import json as _json

            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature,
                        "top_p": kwargs.get("top_p", 0.9),
                        "top_k": kwargs.get("top_k", 40),
                        "repeat_penalty": kwargs.get("repeat_penalty", 1.1),
                    },
                },
                timeout=(10, config.time_out),
                stream=True,
            )

            if response.status_code != 200:
                raise RuntimeError(f"Ollama error {response.status_code}: {response.text}")

            full_response = []
            for raw_line in response.iter_lines():
                if not raw_line:
                    continue
                try:
                    chunk = _json.loads(raw_line)
                    token = chunk.get("response", "")
                    if token:
                        full_response.append(token)
                    if chunk.get("done"):
                        break
                except _json.JSONDecodeError:
                    continue

            return "".join(full_response)

        except requests.exceptions.ConnectionError:
            return "❌ Error: Cannot connect to Ollama server. Make sure it's running."
        except requests.exceptions.Timeout:
            return "⏱️ Error: Request timeout. Try increasing timeout value."
        except Exception as e:
            return f"⚠️ Warning in OllamaInterface: {str(e)}"

    async def generate_async(self, prompt: str, max_tokens: int = config.max_token,
                              temperature: float = config.temperature, **kwargs) -> str:
        try:
            import aiohttp, json as _json
            payload = {
                "model": self.model_name, "prompt": prompt, "stream": True,
                "options": {
                    "num_predict": max_tokens, "temperature": temperature,
                    "top_p": kwargs.get("top_p", 0.9), "top_k": kwargs.get("top_k", 40),
                    "repeat_penalty": kwargs.get("repeat_penalty", 1.1),
                },
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(connect=10, sock_read=config.time_out),
                ) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"Ollama {resp.status}: {await resp.text()}")
                    full_response = []
                    async for raw_line in resp.content:
                        line = raw_line.decode().strip()
                        if not line:
                            continue
                        try:
                            chunk = _json.loads(line)
                            token = chunk.get("response", "")
                            if token:
                                full_response.append(token)
                            if chunk.get("done"):
                                break
                        except _json.JSONDecodeError:
                            continue
                    return "".join(full_response)
        except ImportError:
            return await asyncio.to_thread(self.generate, prompt, max_tokens, temperature, **kwargs)
        except Exception as e:
            return f"⚠️ Ollama async error: {e}"

    async def astream(self, prompt: str, max_tokens: int = config.max_token,
                      temperature: float = config.temperature, **kwargs):
        import json as _json
        try:
            import aiohttp
            payload = {
                "model": self.model_name, "prompt": prompt, "stream": True,
                "options": {"num_predict": max_tokens, "temperature": temperature},
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate", json=payload,
                    timeout=aiohttp.ClientTimeout(total=config.time_out),
                ) as resp:
                    async for raw_line in resp.content:
                        line = raw_line.decode().strip()
                        if line:
                            try:
                                chunk = _json.loads(line)
                                token = chunk.get("response", "")
                                if token:
                                    yield token
                                if chunk.get("done"):
                                    break
                            except _json.JSONDecodeError:
                                continue
        except ImportError:
            response = await asyncio.to_thread(self.generate, prompt, max_tokens, temperature, **kwargs)
            for word in response.split(" "):
                yield word + " "
                await asyncio.sleep(0)
        except Exception as e:
            yield f"⚠️ Ollama stream error: {e}"

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def list_models(self) -> list:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model.get("name", "") for model in models]
            return []
        except Exception as e:
            print(f"⚠️ Error listing models: {e}")
            return []

    def pull_model(self, model_name: str = None) -> bool:
        model = model_name or self.model_name
        try:
            print(f"📥 Downloading model: {model}")
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model},
                stream=True,
                timeout=600,
            )
            if response.status_code == 200:
                print(f"✓ Model {model} downloaded successfully!")
                return True
            else:
                print(f"❌ Failed to download model: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Error downloading model: {e}")
            return False

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def __del__(self):
        # ✅ Safe: only stops server if WE started it
        self.stop_server()