"""JSON Output Parser"""

import json
import re
from typing import Any, Dict, Optional, Type, Union

from .base_parser import BaseOutputParser, ParsingError


class JSONOutputParser(BaseOutputParser):
    """Parse JSON output from text.

    Extracts and validates JSON from raw text, optionally validating
    against a Pydantic model.

    Features:
        - Robust JSON extraction using raw_decode (handles text before/after JSON)
        - Extracts JSON from markdown code blocks
        - Handles malformed JSON with helpful error messages
        - Optional Pydantic model validation
        - Supports JSON objects and arrays

    Example:
        parser = JSONOutputParser()
        result = parser.parse('Here is the data: {"name": "John", "age": 30}')
        # Returns: {'name': 'John', 'age': 30}

        # With Pydantic validation
        from pydantic import BaseModel

        class User(BaseModel):
            name: str
            age: int

        parser = JSONOutputParser(pydantic_model=User)
        user = parser.parse('{"name": "John", "age": 30}')
        # Returns: User(name='John', age=30)
    """

    def __init__(self, pydantic_model: Optional[Type] = None):
        """Initialize JSON parser.

        Args:
            pydantic_model: Optional Pydantic BaseModel class for validation
        """
        self.pydantic_model = pydantic_model
        if pydantic_model is not None:
            self._validate_pydantic_model(pydantic_model)

    def parse(self, text: str) -> Union[Dict[str, Any], list, Any]:
        """Parse JSON from text.

        Args:
            text: Raw text containing JSON

        Returns:
            Parsed JSON as dict, list, or Pydantic model instance

        Raises:
            ParsingError: If JSON cannot be extracted or parsed
        """
        if not text or not text.strip():
            raise ParsingError("Cannot parse empty text", original_text=text)

        json_str = self._extract_json(text)

        if json_str is None:
            raise ParsingError(
                "No valid JSON found in text. Expected a JSON object {...} or array [...].",
                original_text=text,
            )

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ParsingError(
                f"Invalid JSON: {e.msg} at line {e.lineno}, col {e.colno}",
                original_text=json_str,
                cause=e,
            )

        if self.pydantic_model:
            if not isinstance(data, dict):
                raise ParsingError(
                    f"Expected JSON object for Pydantic model, got {type(data).__name__}",
                    original_text=json_str,
                )
            try:
                return self.pydantic_model(**data)
            except Exception as e:
                raise ParsingError(
                    f"JSON does not match schema for {self.pydantic_model.__name__}: {e}",
                    original_text=json_str,
                    cause=e,
                )

        return data

    def get_format_instructions(self) -> str:
        """Get JSON format instructions.

        Returns:
            Format instructions string for LLM
        """
        if self.pydantic_model:
            schema = self.pydantic_model.model_json_schema()
            schema_str = json.dumps(schema, ensure_ascii=False, indent=2)
            return (
                f"Return ONLY valid JSON matching this schema (no extra text):\n"
                f"```json\n{schema_str}\n```\n"
                f"Ensure all required fields are present and types are correct."
            )

        return (
            "Return ONLY valid JSON with no surrounding text.\n"
            'Use double quotes for keys and string values.\n'
            'Example: {"key": "value", "number": 42}'
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_json(self, text: str) -> Optional[str]:
        """Extract the first valid JSON value from text.

        Strategy:
        1. Strip markdown code fences (```json ... ``` or ``` ... ```)
        2. Use json.JSONDecoder.raw_decode to find the first decodable value
           starting from each '{' or '[' character — O(n) in practice.

        Args:
            text: Text that may contain JSON

        Returns:
            Extracted JSON string, or None if not found
        """
        text = text.strip()

        # Strategy 1: markdown code block
        code_fence = re.search(
            r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE
        )
        if code_fence:
            candidate = code_fence.group(1).strip()
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass  # fall through to raw_decode

        # Strategy 2: raw_decode — scan for first '{' or '['
        decoder = json.JSONDecoder()
        for start_char in ('{', '['):
            idx = text.find(start_char)
            while idx != -1:
                try:
                    obj, end_idx = decoder.raw_decode(text, idx)
                    return text[idx:end_idx]
                except json.JSONDecodeError:
                    idx = text.find(start_char, idx + 1)

        return None

    def _validate_pydantic_model(self, model: Type) -> None:
        """Validate that the model is a proper Pydantic BaseModel subclass."""
        try:
            from pydantic import BaseModel

            if not (isinstance(model, type) and issubclass(model, BaseModel)):
                raise ValueError("pydantic_model must be a Pydantic BaseModel subclass")
        except ImportError:
            raise ImportError(
                "Pydantic is required for model validation. "
                "Install it with: pip install pydantic"
            )

    def __repr__(self) -> str:
        model_name = self.pydantic_model.__name__ if self.pydantic_model else "None"
        return f"JSONOutputParser(pydantic_model={model_name})"
