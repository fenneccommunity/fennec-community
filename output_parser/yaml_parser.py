"""YAML Output Parser"""

import re
from typing import Any, Dict, Optional, Type

from .base_parser import BaseOutputParser, ParsingError

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class YAMLOutputParser(BaseOutputParser):
    """Parse YAML-formatted output from LLM text.

    Extracts YAML from raw text and optionally validates the result
    against a Pydantic model.

    Features:
        - Extracts YAML from markdown code blocks (```yaml ... ```)
        - Optional Pydantic model validation
        - Safe loading (no arbitrary code execution)

    Requirements:
        pip install pyyaml

    Example:
        parser = YAMLOutputParser()
        text = '''
        ```yaml
        name: Alice
        age: 30
        skills:
          - Python
          - YAML
        ```
        '''
        result = parser.parse(text)
        # {'name': 'Alice', 'age': 30, 'skills': ['Python', 'YAML']}
    """

    def __init__(self, pydantic_model: Optional[Type] = None):
        """Initialize YAML parser.

        Args:
            pydantic_model: Optional Pydantic BaseModel subclass for validation

        Raises:
            ImportError: If pyyaml is not installed
        """
        if not YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required for YAMLOutputParser. "
                "Install it with: pip install pyyaml"
            )

        self.pydantic_model = pydantic_model

        if pydantic_model is not None:
            self._validate_pydantic_model(pydantic_model)

    def parse(self, text: str) -> Any:
        """Parse YAML from text.

        Args:
            text: Raw text containing YAML

        Returns:
            Parsed YAML content (dict, list, etc.) or Pydantic model instance

        Raises:
            ParsingError: If YAML cannot be extracted or is invalid
        """
        if not text or not text.strip():
            raise ParsingError("Cannot parse empty text", original_text=text)

        yaml_text = self._extract_yaml(text)

        try:
            data = yaml.safe_load(yaml_text)
        except yaml.YAMLError as e:
            raise ParsingError(
                f"Invalid YAML: {e}",
                original_text=yaml_text,
                cause=e,
            )

        if data is None:
            raise ParsingError("YAML parsed to None/empty.", original_text=yaml_text)

        if self.pydantic_model:
            if not isinstance(data, dict):
                raise ParsingError(
                    f"Expected YAML mapping for Pydantic model, got {type(data).__name__}",
                    original_text=yaml_text,
                )
            try:
                return self.pydantic_model(**data)
            except Exception as e:
                raise ParsingError(
                    f"YAML does not match schema for {self.pydantic_model.__name__}: {e}",
                    original_text=yaml_text,
                    cause=e,
                )

        return data

    def get_format_instructions(self) -> str:
        """Return YAML format instructions."""
        if self.pydantic_model:
            schema = self.pydantic_model.model_json_schema()
            properties = schema.get("properties", {})
            required = set(schema.get("required", []))

            lines = [
                f"Return ONLY valid YAML matching the {self.pydantic_model.__name__} schema.",
                "```yaml",
            ]
            for field, info in properties.items():
                ftype = info.get("type", "any")
                req_marker = "  # required" if field in required else "  # optional"
                lines.append(f"{field}: <{ftype}>{req_marker}")
            lines.append("```")
            return "\n".join(lines)

        return (
            "Return ONLY valid YAML (no surrounding text).\n"
            "Wrap in a ```yaml code block.\n\n"
            "Example:\n"
            "```yaml\n"
            "key: value\n"
            "number: 42\n"
            "items:\n"
            "  - item1\n"
            "  - item2\n"
            "```"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_yaml(self, text: str) -> str:
        """
        Extract clean YAML from messy LLM output.
    
        Raises:
            ParsingError: If extraction fails or result is empty/invalid
        """
    
        if not text or not text.strip():
            raise ParsingError("Empty input text; cannot extract YAML.", original_text=text)
    
        try:
            # ─────────────────────────────
            # 1. Extract from code block (BEST CASE)
            # ─────────────────────────────
            match = re.search(r"```(?:yaml|yml)?\s*([\s\S]*?)```", text, re.IGNORECASE)
            if match:
                yaml_text = match.group(1).strip()
                if yaml_text:
                    return yaml_text
                else:
                    raise ParsingError(
                        "YAML code block found but empty.",
                        original_text=text,
                    )
    
            # ─────────────────────────────
            # 2. Remove obvious noise
            # ─────────────────────────────
            lines = text.splitlines()
            cleaned_lines = []
    
            for line in lines:
                stripped = line.strip()
    
                # تجاهل markdown
                if stripped.startswith("```"):
                    continue
                
                # تجاهل yaml لوحدها
                if stripped.lower() in {"yaml", "yml"}:
                    continue
                
                # تجاهل جمل LLM
                if stripped.lower().startswith((
                    "here is",
                    "here's",
                    "the following",
                    "below is",
                    "this is",
                )):
                    continue
                
                cleaned_lines.append(line)
    
            cleaned_text = "\n".join(cleaned_lines).strip()
    
            if not cleaned_text:
                raise ParsingError(
                    "Text became empty after cleaning; no YAML content found.",
                    original_text=text,
                )
    
            # ─────────────────────────────
            # 3. Detect first YAML key
            # ─────────────────────────────
            key_match = re.search(
                r"^[\s\-]*[a-zA-Z0-9_\-]+\s*:\s*",
                cleaned_text,
                re.MULTILINE,
            )
    
            if key_match:
                yaml_text = cleaned_text[key_match.start():].strip()
            else:
                # fallback
                yaml_text = cleaned_text
    
            # ─────────────────────────────
            # 4. Final validation (basic)
            # ─────────────────────────────
            if ":" not in yaml_text:
                raise ParsingError(
                    "Extracted text does not look like valid YAML (missing ':').",
                    original_text=yaml_text,
                )
    
            return yaml_text
    
        except ParsingError:
            # سيبه يعدي زي ما هو
            raise
        
        except Exception as e:
            raise ParsingError(
                f"Unexpected error during YAML extraction: {e}",
                original_text=text,
                cause=e,
            )
    
    def _validate_pydantic_model(self, model: Type) -> None:
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
        return f"YAMLOutputParser(pydantic_model={model_name})"
