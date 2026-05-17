"""
Intelligent retry and regeneration handler.

When parsing fails even after fixing, RetryHandler re-prompts the LLM with
progressively stricter instructions until a valid output is obtained or the
max retry budget is exhausted.

Strategies (applied in order):
    1. STRICT_FORMAT  — Add explicit format instructions and re-ask
    2. JSON_STRICT    — Demand JSON-only output with schema example
    3. SIMPLIFIED     — Ask a stripped-down version of the question
    4. GRACEFUL_FAIL  — Return a structured error payload
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Type

logger = logging.getLogger(__name__)

LLMCallable = Callable[[str], str]


# ─────────────────────────────────────────────
# Retry strategy enum
# ─────────────────────────────────────────────

class RetryStrategy(Enum):
    STRICT_FORMAT  = auto()  # Add format instructions, re-ask
    JSON_STRICT    = auto()  # Force JSON-only mode
    SIMPLIFIED     = auto()  # Stripped-down prompt
    GRACEFUL_FAIL  = auto()  # Return error payload


# ─────────────────────────────────────────────
# Result
# ─────────────────────────────────────────────

@dataclass
class RetryResult:
    """Outcome of a RetryHandler.run() call."""
    success: bool
    response: str                       # Raw LLM response from winning attempt
    attempts: int = 0
    strategy_used: Optional[RetryStrategy] = None
    errors: List[str] = field(default_factory=list)
    total_duration_ms: float = 0.0

    def as_error_payload(self, original_query: str = "") -> Dict[str, Any]:
        """Return a structured error payload when all retries fail."""
        return {
            "error": True,
            "message": "Failed to obtain a valid response after all retries",
            "attempts": self.attempts,
            "original_query": original_query,
            "errors": self.errors,
        }


# ─────────────────────────────────────────────
# Retry handler
# ─────────────────────────────────────────────

class RetryHandler:
    """
    Manages LLM retry and regeneration when output parsing fails.

    Usage::

        def my_llm(prompt: str) -> str:
            # Call your LLM here
            return llm.generate(prompt)

        handler = RetryHandler(llm_fn=my_llm, max_retries=3)
        result = handler.run(
            original_prompt="What is the capital of France?",
            parse_fn=parser.parse,
            format_instructions=parser.get_format_instructions(),
        )
        if result.success:
            parsed = parser.parse(result.response)
    """

    def __init__(
        self,
        llm_fn: LLMCallable,
        max_retries: int = 3,
        backoff_seconds: float = 0.5,
        pydantic_schema: Optional[Type] = None,
        required_fields: Optional[List[str]] = None,
    ):
        """
        Args:
            llm_fn:           Callable(prompt) → raw LLM response string.
            max_retries:      Maximum number of re-attempts (default 3).
            backoff_seconds:  Wait time between retries (default 0.5s).
            pydantic_schema:  Optional Pydantic model to embed schema example in retry prompt.
            required_fields:  Field names used to build JSON example in retry prompt.
        """
        self.llm_fn = llm_fn
        self.max_retries = max(1, max_retries)
        self.backoff_seconds = backoff_seconds
        self.pydantic_schema = pydantic_schema
        self.required_fields = required_fields or []
        self._strategies = [
            RetryStrategy.STRICT_FORMAT,
            RetryStrategy.JSON_STRICT,
            RetryStrategy.SIMPLIFIED,
            RetryStrategy.GRACEFUL_FAIL,
        ]

    def run(
        self,
        original_prompt: str,
        parse_fn: Callable[[str], Any],
        format_instructions: str = "",
        last_error: str = "",
        last_response: str = "",
    ) -> RetryResult:
        """
        Attempt retries until parse_fn succeeds or budget exhausted.

        Args:
            original_prompt:    The user's original request / question.
            parse_fn:           Callable that parses a string; raises on failure.
            format_instructions: Format hint string from the parser.
            last_error:         Error message from the last failed parse.
            last_response:      The last raw LLM response that failed.

        Returns:
            RetryResult
        """
        start = time.time()
        errors: List[str] = []
        attempts = 0

        for attempt_idx in range(self.max_retries):
            strategy = self._strategies[min(attempt_idx, len(self._strategies) - 2)]

            if strategy == RetryStrategy.GRACEFUL_FAIL:
                break

            prompt = self._build_retry_prompt(
                strategy=strategy,
                original_prompt=original_prompt,
                format_instructions=format_instructions,
                last_error=last_error,
                last_response=last_response,
            )

            logger.info(
                "RetryHandler: attempt %d/%d, strategy=%s",
                attempt_idx + 1, self.max_retries, strategy.name,
            )

            try:
                if attempt_idx > 0 and self.backoff_seconds > 0:
                    time.sleep(self.backoff_seconds * attempt_idx)

                raw_response = self.llm_fn(prompt)
                attempts += 1

                # Test if parseable
                parse_fn(raw_response)

                duration = (time.time() - start) * 1000
                logger.info("RetryHandler: success on attempt %d", attempt_idx + 1)
                return RetryResult(
                    success=True,
                    response=raw_response,
                    attempts=attempts,
                    strategy_used=strategy,
                    errors=errors,
                    total_duration_ms=duration,
                )

            except Exception as exc:
                err_msg = f"Attempt {attempt_idx + 1} ({strategy.name}): {exc}"
                errors.append(err_msg)
                last_error = str(exc)
                logger.warning(err_msg)

        duration = (time.time() - start) * 1000
        return RetryResult(
            success=False,
            response="",
            attempts=attempts,
            strategy_used=RetryStrategy.GRACEFUL_FAIL,
            errors=errors,
            total_duration_ms=duration,
        )

    # ─── Prompt builders ─────────────────────────────────────────────────

    def _build_retry_prompt(
        self,
        strategy: RetryStrategy,
        original_prompt: str,
        format_instructions: str,
        last_error: str,
        last_response: str,
    ) -> str:
        if strategy == RetryStrategy.STRICT_FORMAT:
            return self._strict_format_prompt(
                original_prompt, format_instructions, last_error, last_response
            )
        elif strategy == RetryStrategy.JSON_STRICT:
            return self._json_strict_prompt(original_prompt, last_error)
        elif strategy == RetryStrategy.SIMPLIFIED:
            return self._simplified_prompt(original_prompt)
        else:
            return original_prompt

    def _strict_format_prompt(
        self,
        original: str,
        instructions: str,
        error: str,
        bad_response: str,
    ) -> str:
        parts = [
            "Your previous response could not be parsed. Please try again.\n",
        ]
        if error:
            parts.append(f"Error encountered: {error}\n")
        if bad_response:
            preview = bad_response[:300] + ("..." if len(bad_response) > 300 else "")
            parts.append(f"Your previous response (first 300 chars):\n{preview}\n")
        if instructions:
            parts.append(f"Format requirements:\n{instructions}\n")
        parts.append(f"\nOriginal question:\n{original}")
        return "\n".join(parts)

    def _json_strict_prompt(self, original: str, error: str) -> str:
        schema_example = self._build_json_example()
        return (
            f"IMPORTANT: You MUST respond with ONLY valid JSON. No explanation, "
            f"no markdown, no surrounding text — just the JSON object.\n\n"
            f"Previous parse error: {error}\n\n"
            f"Required JSON format:\n{schema_example}\n\n"
            f"Question: {original}\n\n"
            f"Respond with ONLY the JSON object:"
        )

    def _simplified_prompt(self, original: str) -> str:
        return (
            f"Please answer the following question concisely and return your answer "
            f"as a JSON object with an 'answer' key.\n\n"
            f"Question: {original}\n\n"
            f'Example: {{"answer": "Your answer here", "confidence": 0.9}}'
        )

    def _build_json_example(self) -> str:
        """Build a minimal JSON example from schema or required_fields."""
        if self.pydantic_schema:
            try:
                schema = self.pydantic_schema.model_json_schema()
                props = schema.get("properties", {})
                example: Dict[str, Any] = {}
                type_defaults = {
                    "string": "example text",
                    "integer": 0,
                    "number": 0.0,
                    "boolean": True,
                    "array": [],
                    "object": {},
                }
                for key, info in props.items():
                    t = info.get("type", "string")
                    example[key] = type_defaults.get(t, None)
                return json.dumps(example, indent=2)
            except Exception:
                pass

        if self.required_fields:
            example = {f: f"<{f}_value>" for f in self.required_fields}
            return json.dumps(example, indent=2)

        return '{\n  "answer": "...",\n  "confidence": 0.9\n}'


# ─────────────────────────────────────────────
# Graceful degradation helper
# ─────────────────────────────────────────────

def graceful_fallback(
    raw_text: str,
    query: str = "",
) -> Dict[str, Any]:
    """
    Last-resort fallback when all parsing and retry attempts fail.
    Returns a structured error dict that is always safe to return to callers.
    """
    logger.error("All parsing attempts failed. Returning graceful fallback for query: %r", query[:80])
    return {
        "answer": raw_text.strip() if raw_text else "Unable to parse response",
        "sources": [],
        "confidence": 0.0,
        "_parse_error": True,
        "_raw": raw_text[:500] if raw_text else "",
        "_query": query[:200] if query else "",
    }
