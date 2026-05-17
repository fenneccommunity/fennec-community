from .base_llm_interface import BaseLLMInterface
from .config_llm import llm_config
from typing import Optional
import random
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
config = llm_config()


class GeminiInterface(BaseLLMInterface):
    """Google Gemini interface """
    
    def __init__(self, model_name = config.gemini_model, api_key: Optional[str] = None, **kwargs):
        """
        Initialize Gemini Interface 
        
        Args:
            model_name: Model name (prefer gemini-2.0-flash-exp for longer responses) 
            api_key: API key
            **kwargs: Additional settings
        """
        super().__init__(model_name, api_key, **kwargs)
        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                " Google Gemini Not Found!\n"
                "Install: pip install google-genai"
            )
        
        # ✅ Better default settings for Arabic text - إعدادات افتراضية أفضل للنصوص العربية
        self.default_config = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "max_output_tokens": config.max_token,  # ✅ Increase maximum limit - زيادة الحد الأقصى
        }
        
        # ✅ Rate limiting to avoid 429 errors (Gemini free tier: 15 req/min)
        # تحديد معدل الطلبات لتجنب خطأ 429 (الطبقة المجانية: 15 طلب/دقيقة)
        # Conservative: 5s gap = max 12 req/min, well under the 15 req/min limit
        # even when ProtectedLLMInterface fires multiple retries per question.
        self._min_request_interval = 5.0  # seconds between requests - ثواني بين الطلبات
        self._last_request_time = 0.0

    def generate(
        self,
        prompt: str,
        max_tokens: int = config.max_token,  # ✅ None instead of small default value - None بدلاً من القيمة الافتراضية الصغيرة
        temperature: float = None,
        **kwargs
    ) -> str:
        """
        Generate text with automatic retry on errors 
        
        Args:
            prompt: Input text 
            max_tokens: Maximum tokens (None = use large default value) 
            temperature: Randomness degree 
            **kwargs: Additional settings 
        
        Returns:
            Generated text 
        """
        # ✅ Rate limiting: ensure minimum interval between requests
        # تحديد معدل الطلبات: ضمان الحد الأدنى من الفاصل الزمني بين الطلبات
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

        max_retries = 5  # ✅ Increase retry attempts - زيادة عدد المحاولات
        base_delay = 5  # ✅ Longer delay - تأخير أطول
        
        # ✅ Use improved default values - استخدام القيم الافتراضية المحسّنة
        generation_config = {
            "temperature": temperature if temperature is not None else self.default_config["temperature"],
            "max_output_tokens": max_tokens if max_tokens is not None else self.default_config["max_output_tokens"],
            "top_p": kwargs.get("top_p", self.default_config["top_p"]),
            "top_k": kwargs.get("top_k", self.default_config["top_k"]),
        }
        
        # ✅ Remove custom keys from kwargs - إزالة المفاتيح المخصصة من kwargs
        for key in ["top_p", "top_k"]:
            kwargs.pop(key, None)
        
        generation_config.update(kwargs)

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=generation_config
                )

                # ✅ Extract text from response - استخراج النص من الاستجابة
                text_parts = []

                if hasattr(response, "candidates"):
                    for candidate in response.candidates:
                        content = getattr(candidate, "content", None)
                        if content and hasattr(content, "parts"):
                            for part in content.parts:
                                if hasattr(part, "text"):
                                    text_parts.append(part.text)

                if text_parts:
                    result = "".join(text_parts).strip()
                    # ✅ Check that response is not too short - تحقق من أن الإجابة ليست قصيرة جداً
                    # Only retry when result is truly empty (< 3 chars).
                    # Short factual answers like "4", "Berlin", "Shakespeare"
                    # are valid — do NOT re-request them.
                    if len(result) < 3 and attempt < max_retries - 1:
                        print(f"⚠️ Response too short ({len(result)} characters), retrying...")
                        time.sleep(1)
                        continue
                    return result

                # Fallback - البديل
                if hasattr(response, "text"):
                    return response.text.strip()

                # ✅ If we didn't get any text - إذا لم نحصل على نص
                if attempt < max_retries - 1:
                    print(f"⚠️ No response received, attempt {attempt + 1}/{max_retries}")
                    time.sleep(base_delay)
                    continue

                return ""

            except Exception as e:
                error_msg = str(e)

                # ✅ Handle 503/UNAVAILABLE errors - معالجة أخطاء 503/UNAVAILABLE
                if "503" in error_msg or "UNAVAILABLE" in error_msg or "429" in error_msg:
                    if attempt == max_retries - 1:
                        print(f"❌ Failed after {max_retries} attempts: {error_msg}")
                        break

                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1.5)
                    print(
                        f"⚠️ Gemini error ({error_msg[:50]}...). "
                        f"Retry {attempt + 1}/{max_retries} after {delay:.1f}s... - "
                    )
                    time.sleep(delay)
                    # ✅ Reset rate-limit timer after sleeping so the next attempt
                    # is not throttled on top of the backoff delay already served.
                    self._last_request_time = time.time()
                    continue

                # ✅ Other errors - أخطاء أخرى
                return f"Error: {error_msg}"

        return "❌ Error: All attempts failed with Gemini"
    async def generate_async(
        self,
        prompt: str,
        max_tokens: int = None,
        temperature: float = None,
        **kwargs
    ) -> str:
        """Generate text asynchronously"""
    
        import asyncio
    
        try:
            # ✅ Rate limiting for async path as well
            elapsed = time.time() - self._last_request_time
            if elapsed < self._min_request_interval:
                await asyncio.sleep(self._min_request_interval - elapsed)
            self._last_request_time = time.time()

            generation_config = {
                "temperature": temperature if temperature is not None else self.default_config["temperature"],
                "max_output_tokens": max_tokens if max_tokens is not None else self.default_config["max_output_tokens"],
                "top_p": kwargs.get("top_p", self.default_config["top_p"]),
                "top_k": kwargs.get("top_k", self.default_config["top_k"]),
            }
    
            for key in ["top_p", "top_k"]:
                kwargs.pop(key, None)
    
            generation_config.update(kwargs)
    
            loop = asyncio.get_running_loop()
    
            response = await loop.run_in_executor(
                None,
                lambda: self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=generation_config
                )
            )
    
            text_parts = []
    
            if hasattr(response, "candidates"):
                for candidate in response.candidates:
                    content = getattr(candidate, "content", None)
                    if content and hasattr(content, "parts"):
                        for part in content.parts:
                            if hasattr(part, "text"):
                                text_parts.append(part.text)
    
            if text_parts:
                return "".join(text_parts).strip()
    
            return getattr(response, "text", "").strip()
    
        except Exception as e:
            print(f"❌ Gemini async error: {e}")
            return f"Error: {str(e)}"


    async def astream(self, prompt: str, max_tokens: int = None,
                      temperature: float = None, **kwargs):
        """Stream tokens from Gemini asynchronously."""
        import asyncio
        # Gemini SDK may not support true streaming yet; fallback to word-by-word
        response = await self.generate_async(prompt, max_tokens, temperature, **kwargs)
        for word in response.split(" "):
            yield word + " "
            await asyncio.sleep(0)