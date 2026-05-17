from typing import Optional, AsyncIterator
import asyncio
from abc import ABC, abstractmethod
from .config_llm import llm_config

llm_config_ = llm_config()


class BaseLLMInterface(ABC):
    """Main Interface For All LLM Support\n"""

    def __init__(self, model_name: str, api_key: str, **kwargs):
        self.model_name = model_name
        self.api_key = api_key
        self.kwargs = kwargs

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = llm_config_.max_token,
        temperature: float = llm_config_.temperature,
        **kwargs,
    ) -> str:
        """generate text"""
        pass

    @abstractmethod
    async def generate_async(
        self,
        prompt: str,
        max_tokens: int = llm_config_.max_token,
        temperature: float = llm_config_.temperature,
        **kwargs,
    ) -> str:
        """generate text as async"""
        pass

    async def astream(
        self,
        prompt: str,
        max_tokens: int = llm_config_.max_token,
        temperature: float = llm_config_.temperature,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        Stream tokens one by one (async generator).
        Default: generates full response then yields word-by-word.
        Override in subclasses that support native streaming.

        Example:
            async for token in llm.astream("مرحباً"):
                print(token, end="", flush=True)
        """
        response = await self.generate_async(prompt, max_tokens, temperature, **kwargs)
        for word in response.split(" "):
            yield word + " "
            await asyncio.sleep(0)  # yield control to event loop

    # ── Async context manager ────────────────────────────────────────
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.acleanup()
        return False

    async def acleanup(self):
        """Override in subclasses to release async resources (e.g., aiohttp sessions)"""
        pass

    def validate_connection(
        self,
        test_prompt: str = "test",
        max_tokens: int = 10,
        temperature: float = 0.7,
        async_mode: bool = False,
    ) -> dict:
        """
        Check LLM connection and response.

        Returns:
            dict: {
                "success": bool,
                "reason": str,
                "response": str (if success)
            }
        """
        try:
            if async_mode:
                import asyncio
                response = asyncio.run(
                    self.generate_async(test_prompt, max_tokens=max_tokens, temperature=temperature)
                )
            else:
                response = self.generate(test_prompt, max_tokens=max_tokens, temperature=temperature)

            if response:
                return {"success": True, "reason": "Connection successful", "response": response}
            else:
                return {"success": False, "reason": "Empty response", "response": None}

        except Exception as e:
            return {"success": False, "reason": str(e), "response": None}

    