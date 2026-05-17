"""
Anthropic (Claude) Interface — sync + full async + streaming
"""
from __future__ import annotations

import logging
from typing import AsyncIterator, Optional, Union

from .base_llm_interface import BaseLLMInterface
from .config_llm import llm_config

config = llm_config()
logger = logging.getLogger(__name__)


class AnthropicInterface(BaseLLMInterface):
    """Anthropic (Claude) — sync & async interface with native streaming"""

    def __init__(
        self,
        model_name: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model_name, api_key, **kwargs)
        try:
            import anthropic
            self._sync_client  = anthropic.Anthropic(api_key=self.api_key)
            self._async_client = anthropic.AsyncAnthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("install anthropic: pip install anthropic")

    # ── helpers ─────────────────────────────────────────────────────
    @staticmethod
    def _build_messages(
        prompt: Optional[str],
        messages: Optional[list[dict]],
    ) -> list[dict]:
        """Resolve prompt / messages into a messages list (mirrors OpenAI interface)."""
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
        max_tokens: int = config.max_token,
        temperature: float = config.temperature,
        **kwargs,
    ) -> Union[str, dict]:
        """Generate text synchronously."""
        try:
            msgs = self._build_messages(prompt, messages)
            response = self._sync_client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=msgs,
                **kwargs,
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic sync error: {e}")
            return {"error": str(e)}

    # ── Async ───────────────────────────────────────────────────────
    async def generate_async(
        self,
        messages: Optional[list[dict]] = None,
        prompt: Optional[str] = None,
        max_tokens: int = config.max_token,
        temperature: float = config.temperature,
        **kwargs,
    ) -> Union[str, dict]:
        """Generate text asynchronously."""
        try:
            msgs = self._build_messages(prompt, messages)
            response = await self._async_client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=msgs,
                **kwargs,
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic async error: {e}")
            return {"error": str(e)}

    # ── Native token streaming ───────────────────────────────────────
    async def astream(
        self,
        messages: Optional[list[dict]] = None,
        prompt: Optional[str] = None,
        max_tokens: int = config.max_token,
        temperature: float = config.temperature,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream tokens natively from Claude."""
        try:
            msgs = self._build_messages(prompt, messages)
            async with self._async_client.messages.stream(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=msgs,
                **kwargs,
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as e:
            logger.error(f"Anthropic stream error: {e}")
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