"""Structured Output Parser"""

import re
from typing import Any, Dict, List, Optional
from .base_parser import BaseOutputParser, ParsingError


class ResponseSchema:
    """Schema definition for a single structured output field.

    Attributes:
        name        : Field name (used in extraction patterns)
        description : Human-readable description shown in format instructions
        type        : Expected data type – one of: str, int, float, bool, list
        required    : Whether this field must be present (default: True)
        aliases     : Alternative names the LLM might use for this field
    """

    VALID_TYPES = {"str", "int", "float", "bool", "list"}

    def __init__(
        self,
        name: str,
        description: str,
        type: str = "str",
        required: bool = True,
        aliases: Optional[List[str]] = None,
    ):
        if type not in self.VALID_TYPES:
            raise ValueError(
                f"Unsupported type '{type}'. Must be one of: {self.VALID_TYPES}"
            )
        self.name = name
        self.description = description
        self.type = type
        self.required = required
        self.aliases: List[str] = aliases or []

    @property
    def all_names(self) -> List[str]:
        """All names that should match this field (primary + aliases)."""
        return [self.name] + self.aliases

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "type": self.type,
            "required": self.required,
            "aliases": self.aliases,
        }

    def __repr__(self) -> str:
        return f"ResponseSchema(name={self.name!r}, type={self.type!r})"


class StructuredOutputParser(BaseOutputParser):
    """Parse structured key-value output from text.

    Extracts specific named fields from free-form text using flexible
    pattern matching.

    Supported input formats:
        field_name: value
        field_name = value
        **field_name**: value  (markdown bold)
        <field_name>value</field_name>  (XML-style)

    Features:
        - Type coercion (str, int, float, bool, list)
        - Required field validation
        - Case-insensitive field matching
        - Alias support (alternative field names)
        - Multi-line value support

    Example:
        schemas = [
            ResponseSchema("answer", "The answer to the question"),
            ResponseSchema("confidence", "Confidence score 0–1", type="float"),
            ResponseSchema("sources", "List of sources", type="list", required=False),
        ]

        parser = StructuredOutputParser(response_schemas=schemas)

        text = '''
        answer: The capital of France is Paris
        confidence: 0.95
        sources: Wikipedia, Encyclopedia Britannica
        '''

        result = parser.parse(text)
        # {'answer': 'The capital of France is Paris',
        #  'confidence': 0.95,
        #  'sources': ['Wikipedia', 'Encyclopedia Britannica']}
    """

    def __init__(self, response_schemas: List[ResponseSchema]):
        if not response_schemas:
            raise ValueError("response_schemas cannot be empty")

        self.response_schemas = response_schemas

        # Build a lookup: all lowercase names/aliases → schema
        self._schema_map: Dict[str, ResponseSchema] = {}
        for schema in response_schemas:
            for name in schema.all_names:
                key = name.lower()
                if key in self._schema_map:
                    raise ValueError(
                        f"Duplicate field name or alias: '{key}'"
                    )
                self._schema_map[key] = schema

    def parse(self, text: str) -> Dict[str, Any]:
        """Parse structured output from text.

        Args:
            text: Raw text containing structured fields

        Returns:
            Dictionary with extracted and typed field values

        Raises:
            ParsingError: If required fields are missing or type conversion fails
        """
        if not text or not text.strip():
            raise ParsingError("Cannot parse empty text", original_text=text)

        result: Dict[str, Any] = {}

        for schema in self.response_schemas:
            raw_value = self._extract_field(text, schema)

            if raw_value is None:
                if schema.required:
                    raise ParsingError(
                        f"Required field '{schema.name}' not found in text.",
                        original_text=text,
                    )
                continue

            try:
                result[schema.name] = self._convert_type(raw_value, schema.type)
            except ValueError as e:
                raise ParsingError(
                    f"Cannot convert field '{schema.name}' value {raw_value!r} "
                    f"to type '{schema.type}': {e}",
                    original_text=text,
                    cause=e,
                )

        return result

    def get_format_instructions(self) -> str:
        lines = ["Return the result using the following key: value format:\n"]
        for schema in self.response_schemas:
            req = " (required)" if schema.required else " (optional)"
            lines.append(f"{schema.name}: {schema.description}{req}")

        lines.append("\nExample:")
        for schema in self.response_schemas:
            example = self._get_example_value(schema.type)
            lines.append(f"{schema.name}: {example}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_field(self, text: str, schema: ResponseSchema) -> Optional[str]:
        """Extract a field value from text by trying multiple patterns.

        The function tries every name in schema.all_names and four different
        formats in order of specificity.
        """
        for name in schema.all_names:
            escaped = re.escape(name)

            patterns = [
                # XML / tag style:  <field>value</field>
                rf"<{escaped}>\s*(.*?)\s*</{escaped}>",
                # Markdown bold:  **field**: value
                rf"\*\*{escaped}\*\*\s*[:\-=]\s*(.+?)(?=\n\s*\*\*|\Z)",
                # colon separator:  field: value
                rf"(?m)^{escaped}\s*:\s*(.+?)(?=\n\s*\w[\w\s]*\s*:|$)",
                # equals separator:  field = value
                rf"(?m)^{escaped}\s*=\s*(.+?)(?=\n\s*\w[\w\s]*\s*=|$)",
            ]

            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match:
                    value = match.group(1).strip()
                    # Collapse internal whitespace (handles multi-line values)
                    value = re.sub(r"\s+", " ", value)
                    return value if value else None

        return None

    def _convert_type(self, value: str, target_type: str) -> Any:
        """Convert a string value to the target Python type."""
        value = value.strip()

        if target_type == "str":
            return value

        if target_type == "int":
            match = re.search(r"-?\d+", value)
            if match:
                return int(match.group())
            raise ValueError(f"No integer found in {value!r}")

        if target_type == "float":
            match = re.search(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?", value)
            if match:
                return float(match.group())
            raise ValueError(f"No float found in {value!r}")

        if target_type == "bool":
            lower = value.lower()
            if lower in {"true", "yes", "1", "correct", "y", "on"}:
                return True
            if lower in {"false", "no", "0", "incorrect", "n", "off"}:
                return False
            raise ValueError(f"Cannot interpret {value!r} as bool")

        if target_type == "list":
            # Try splitting by comma, semicolon, pipe, or newline
            raw_items = re.split(r"[,;|\n]", value)
            items = [i.strip() for i in raw_items if i.strip()]
            # Remove surrounding quotes from each item
            items = [re.sub(r"^['\"]|['\"]$", "", i) for i in items]
            return items

        raise ValueError(f"Unsupported type: {target_type!r}")

    @staticmethod
    def _get_example_value(type_name: str) -> str:
        return {
            "str": "example text",
            "int": "42",
            "float": "0.95",
            "bool": "true",
            "list": "item1, item2, item3",
        }.get(type_name, "value")

    def __repr__(self) -> str:
        field_names = [s.name for s in self.response_schemas]
        return f"StructuredOutputParser(fields={field_names})"
