"""Mistral AI Interface — sync + full async + native streaming"""
from __future__ import annotations
from typing import AsyncIterator, Optional
from .base_llm_interface import BaseLLMInterface
from .config_llm import llm_config

config = llm_config()


class MistralInterface(BaseLLMInterface):
    """
    Mistral AI — sync & async interface with native streaming.

    Supported models:
        - mistral-small-latest       
        - mistral-medium-latest      
        - mistral-large-latest       
        - open-mistral-7b            
        - open-mixtral-8x7b          
        - open-mixtral-8x22b         
        - codestral-latest           

    Required:
        pip install mistralai

    Example:
        >>> llm = MistralInterface(
        ...     model_name="mistral-large-latest",
        ...     api_key="your-api-key",
        ... )
        >>> response = llm.generate("اشرح مفهوم RAG في جملتين.")
        >>> print(response)

        # async:
        >>> result = await llm.generate_async("ما هو الفرق بين GPT و Mistral؟")

        # streaming:
        >>> async for token in llm.astream("اكتب قصيدة عن الذكاء الاصطناعي"):
        ...     print(token, end="", flush=True)
    """

    def __init__(
        self,
        model_name: str = config.mistral_model,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model_name, api_key, **kwargs)
        try:
            from mistralai import Mistral
            self._client = Mistral(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "mistralai library is required. Install it using: pip install mistralai"
            )

    # ── Sync ─────────────────────────────────────────────────────────
    def generate(
        self,
        prompt: str,
        max_tokens: int = config.max_token,
        temperature: float = config.temperature,
        **kwargs,
    ) -> str:
        try:
            response = self._client.chat.complete(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ Mistral error: {e}")
            return f"Error: {e}"

    # ── Async ────────────────────────────────────────────────────────
    async def generate_async(
        self,
        prompt: str,
        max_tokens: int = config.max_token,
        temperature: float = config.temperature,
        **kwargs,
    ) -> str:
        try:
            response = await self._client.chat.complete_async(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {e}"

    # ── Native streaming ─────────────────────────────────────────────
    async def astream(
        self,
        prompt: str,
        max_tokens: int = config.max_token,
        temperature: float = config.temperature,
        **kwargs,
         ) -> AsyncIterator[str]:
        """Stream tokens natively from Mistral AI."""
        try:
            # ✅ await أولاً ثم iterate مباشرة — بدون async with
            stream = await self._client.chat.stream_async(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
            async for chunk in stream:
                delta = chunk.data.choices[0].delta.content
                if delta:
                    yield delta
    
        except Exception as e:
            # Fallback: توليد عادي وتقسيم كلمة كلمة
            try:
                response = await self.generate_async(prompt, max_tokens, temperature, **kwargs)
                for word in response.split(" "):
                    yield word + " "
            except Exception as e2:
                yield f"Error: {e2}"
    
    async def acleanup(self):
        """Close the async client session."""
        try:
            await self._client.close()
        except Exception:
            pass
