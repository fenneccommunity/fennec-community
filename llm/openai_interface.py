from __future__ import annotations

import logging
from typing import AsyncIterator, Optional, Union

from .base_llm_interface import BaseLLMInterface
from .config_llm import llm_config

llm_config_ = llm_config()
logger = logging.getLogger(__name__)


class OpenAIInterface(BaseLLMInterface):
    """OpenAI GPT — sync & async interface with native streaming"""

    def __init__(
        self,
        model_name: str = "gpt-4",
        api_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model_name, api_key, **kwargs)
        try:
            import openai
            self._sync_client  = openai.OpenAI(api_key=self.api_key)
            self._async_client = openai.AsyncOpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("install openai: pip install openai")

    # ── helpers ─────────────────────────────────────────────────────
    @staticmethod
    def _build_messages(
        prompt: Optional[str],
        messages: Optional[list[dict]],
    ) -> list[dict]:
        """Resolve prompt / messages into a messages list."""
        if messages is not None:
            return messages
        if prompt is not None:
            return [{"role": "user", "content": prompt}]
        raise ValueError("Either 'prompt' or 'messages' must be provided.")

    # ── Sync ────────────────────────────────────────────────────────
    def generate(
        self,
        messages: Optional[list[dict]] = None,
        prompt: Optional[str] = None,
        max_tokens: int = llm_config_.max_token,
        temperature: float = llm_config_.temperature,
        **kwargs,
    ) -> Union[str, dict]:
        """Generate text synchronously."""
        try:
            msgs = self._build_messages(prompt, messages)
            r = self._sync_client.chat.completions.create(
                model=self.model_name,
                messages=msgs,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
            return r.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI sync error: {e}")
            return {"error": str(e)}

    # ── Async ───────────────────────────────────────────────────────
    async def generate_async(
        self,
        messages: Optional[list[dict]] = None,
        prompt: Optional[str] = None,
        max_tokens: int = llm_config_.max_token,
        temperature: float = llm_config_.temperature,
        **kwargs,
    ) -> Union[str, dict]:
        """Generate text asynchronously."""
        try:
            msgs = self._build_messages(prompt, messages)
            r = await self._async_client.chat.completions.create(
                model=self.model_name,
                messages=msgs,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
            return r.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI async error: {e}")
            return {"error": str(e)}

    # ── Native streaming ─────────────────────────────────────────────
    async def astream(
        self,
        messages: Optional[list[dict]] = None,
        prompt: Optional[str] = None,
        max_tokens: int = llm_config_.max_token,
        temperature: float = llm_config_.temperature,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream tokens natively from OpenAI."""
        try:
            msgs = self._build_messages(prompt, messages)
            stream = await self._async_client.chat.completions.create(
                model=self.model_name,
                messages=msgs,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                **kwargs,
            )
            async for chunk in stream:
                delta = getattr(chunk.choices[0].delta, "content", None)
                if delta:
                    yield delta
        except Exception as e:
            logger.error(f"OpenAI stream error: {e}")
            yield f"Error: {e}"

    # ── Cleanup ──────────────────────────────────────────────────────
    def cleanup(self):
        """Close sync client."""
        try:
            if self._sync_client:
                self._sync_client.close()
        except Exception as e:
            logger.warning(f"Sync client cleanup error: {e}")

    async def acleanup(self):
        """Close async client."""
        try:
            if self._async_client:
                await self._async_client.close()
        except Exception as e:
            logger.warning(f"Async client cleanup error: {e}")