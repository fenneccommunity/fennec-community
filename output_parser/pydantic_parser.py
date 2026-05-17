"""Pydantic Output Parser"""

import json
from typing import Any, Dict, Type, TypeVar

from .base_parser import BaseOutputParser, ParsingError
from .json_parser import JSONOutputParser

try:
    from pydantic import BaseModel

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = None  # type: ignore[assignment,misc]

T = TypeVar("T")


class PydanticOutputParser(BaseOutputParser):
    """Parse LLM output into a validated Pydantic model instance.

    Combines robust JSON extraction with full Pydantic validation,
    including type coercion, nested models, and custom validators.

    Example:
        from pydantic import BaseModel, Field

        class Product(BaseModel):
            name: str = Field(..., description="Product name")
            price: float = Field(..., ge=0, description="Price in USD")
            in_stock: bool = Field(default=True)

        parser = PydanticOutputParser(pydantic_object=Product)
        text = '{\"name\": \"Widget\", \"price\": 9.99}'
        product = parser.parse(text)
        # Product(name='Widget', price=9.99, in_stock=True)
    """

    def __init__(self, pydantic_object: Type[T]):
        """Initialize Pydantic parser.

        Args:
            pydantic_object: A Pydantic BaseModel subclass to parse into

        Raises:
            ImportError: If pydantic is not installed
            TypeError: If pydantic_object is not a BaseModel subclass
        """
        if not PYDANTIC_AVAILABLE:
            raise ImportError(
                "Pydantic is required for PydanticOutputParser. "
                "Install it with: pip install pydantic"
            )

        if not (isinstance(pydantic_object, type) and issubclass(pydantic_object, BaseModel)):
            raise TypeError(
                f"pydantic_object must be a Pydantic BaseModel subclass. "
                f"Got: {type(pydantic_object)}"
            )

        self.pydantic_object = pydantic_object
        self._json_parser = JSONOutputParser(pydantic_model=pydantic_object)

    def parse(self, text: str) -> T:
        """Parse text into a Pydantic model instance.

        Args:
            text: Raw text containing JSON data

        Returns:
            Validated instance of pydantic_object

        Raises:
            ParsingError: If parsing or validation fails
        """
        try:
            return self._json_parser.parse(text)
        except ParsingError:
            raise
        except Exception as e:
            raise ParsingError(
                f"Failed to parse into {self.pydantic_object.__name__}: {e}",
                original_text=text,
                cause=e,
            )

    def get_format_instructions(self) -> str:
        """Return detailed format instructions including the JSON schema."""
        schema = self.pydantic_object.model_json_schema()
        properties = schema.get("properties", {})
        required_fields = set(schema.get("required", []))

        lines = [
            f"Return ONLY valid JSON that matches the {self.pydantic_object.__name__} schema.",
            "",
            "Fields:",
        ]

        for field_name, field_info in properties.items():
            field_type = self._resolve_type(field_info)
            description = field_info.get("description", "—")
            req = "required" if field_name in required_fields else "optional"
            lines.append(f"  • {field_name} ({field_type}, {req}): {description}")

        lines += ["", "Example:", self._generate_example()]
        return "\n".join(lines)

    def get_schema(self) -> Dict[str, Any]:
        """Return the raw JSON schema of the Pydantic model."""
        return self.pydantic_object.model_json_schema()

    def validate(self, data: Dict[str, Any]) -> T:
        """Validate a plain dictionary against the schema.

        Args:
            data: Dictionary to validate

        Returns:
            Validated model instance

        Raises:
            ParsingError: If validation fails
        """
        try:
            return self.pydantic_object(**data)
        except Exception as e:
            raise ParsingError(
                f"Validation failed for {self.pydantic_object.__name__}: {e}",
                original_text=str(data),
                cause=e,
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_type(field_info: Dict[str, Any]) -> str:
        """Return a human-readable type name from a JSON schema field."""
        if "type" in field_info:
            return field_info["type"]
        if "anyOf" in field_info:
            types = [t.get("type", "?") for t in field_info["anyOf"] if t.get("type") != "null"]
            return " | ".join(types) if types else "any"
        if "$ref" in field_info:
            return field_info["$ref"].split("/")[-1]
        return "any"

    def _generate_example(self) -> str:
        """Generate a minimal example JSON from the model schema."""
        schema = self.pydantic_object.model_json_schema()
        properties = schema.get("properties", {})

        EXAMPLES: Dict[str, Any] = {
            "string": "example",
            "integer": 1,
            "number": 1.0,
            "boolean": True,
            "array": [],
            "object": {},
        }

        example: Dict[str, Any] = {}
        for field_name, field_info in properties.items():
            ftype = self._resolve_type(field_info)
            example[field_name] = EXAMPLES.get(ftype, None)

        return json.dumps(example, indent=2, ensure_ascii=False)

    def __repr__(self) -> str:
        return f"PydanticOutputParser(model={self.pydantic_object.__name__})"
