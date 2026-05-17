"""
schema.py — Typed schemas, enums, and output models for the RAG Output Parser Engine.

All public types used across the engine are defined here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Type, Union

try:
    from pydantic import BaseModel, Field, field_validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object  # type: ignore[assignment,misc]


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class OutputFormat(Enum):
    """Detected or expected format of an LLM output."""
    JSON = "json"
    YAML = "yaml"
    CSV = "csv"
    MARKDOWN_TABLE = "markdown_table"
    NUMBERED_LIST = "numbered_list"
    BULLETED_LIST = "bulleted_list"
    KEY_VALUE = "key_value"
    XML = "xml"
    TOOL_CALL = "tool_call"
    PLAIN_TEXT = "plain_text"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class ParseMode(Enum):
    """Parsing mode that controls strictness and fallback behaviour."""
    STRICT = "strict"           # Fail fast on any deviation
    LENIENT = "lenient"         # Auto-fix minor issues
    SEMANTIC = "semantic"       # Use LLM to extract meaning from freeform text
    TOOL_CALL = "tool_call"     # Parse function/tool invocation syntax


class ValidationStatus(Enum):
    """Result status of a validation check."""
    PASSED = auto()
    FAILED = auto()
    WARNING = auto()
    SKIPPED = auto()


class FixStrategy(Enum):
    """Strategy used by OutputFixer."""
    REGEX_REPAIR = "regex_repair"
    LLM_REFORMAT = "llm_reformat"
    FIELD_INJECTION = "field_injection"
    FALLBACK_PARSE = "fallback_parse"
    NONE = "none"


# ─────────────────────────────────────────────
# Core dataclasses
# ─────────────────────────────────────────────

@dataclass
class ValidationResult:
    """Result of a single validation check."""
    status: ValidationStatus
    field: Optional[str] = None
    message: str = ""
    value: Any = None

    @property
    def passed(self) -> bool:
        return self.status == ValidationStatus.PASSED

    def __repr__(self) -> str:
        return f"ValidationResult({self.status.name}, field={self.field!r}, msg={self.message!r})"


@dataclass
class ParseTrace:
    """Full audit trail for a single parse operation."""
    raw_input: str
    detected_format: OutputFormat = OutputFormat.UNKNOWN
    parse_mode: ParseMode = ParseMode.LENIENT
    fix_applied: FixStrategy = FixStrategy.NONE
    retries: int = 0
    validations: List[ValidationResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    success: bool = False

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def add_validation(self, result: ValidationResult) -> None:
        self.validations.append(result)

    def summary(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "detected_format": self.detected_format.value,
            "parse_mode": self.parse_mode.value,
            "fix_applied": self.fix_applied.value,
            "retries": self.retries,
            "errors": self.errors,
            "warnings": self.warnings,
            "validation_passed": sum(1 for v in self.validations if v.passed),
            "validation_failed": sum(1 for v in self.validations if not v.passed),
            "duration_ms": round(self.duration_ms, 2),
        }


@dataclass
class ParseResult:
    """Final result returned by the parser to callers."""
    data: Any                          # Parsed, validated, typed output
    schema_type: Optional[type] = None # Python type of `data` (if typed)
    trace: Optional[ParseTrace] = None # Full audit trail
    raw: str = ""                      # Original raw LLM text

    @property
    def ok(self) -> bool:
        return self.data is not None

    def as_typed(self, schema_cls: Type) -> Any:
        """Cast `data` into a typed schema class if it's a plain dict."""
        if isinstance(self.data, schema_cls):
            return self.data
        if isinstance(self.data, dict):
            return schema_cls(**self.data)
        raise TypeError(f"Cannot cast {type(self.data)} to {schema_cls}")

    def __repr__(self) -> str:
        dtype = type(self.data).__name__
        return f"ParseResult(ok={self.ok}, type={dtype})"


# ─────────────────────────────────────────────
# Built-in Pydantic Schemas
# ─────────────────────────────────────────────

if PYDANTIC_AVAILABLE:
    class AnswerSchema(BaseModel):
        """Standard RAG answer schema."""
        answer: str = Field(..., description="The answer to the question")
        sources: List[str] = Field(default_factory=list, description="Source references")
        confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score 0-1")
        reasoning: Optional[str] = Field(None, description="Optional chain of thought")

        @field_validator("confidence", mode="before")
        @classmethod
        def _clamp_confidence(cls, v: Any) -> float:
            try:
                f = float(v)
                return max(0.0, min(1.0, f))
            except (TypeError, ValueError):
                return 1.0

    class ToolCallSchema(BaseModel):
        """Schema for LLM tool/function call outputs."""
        tool_name: str = Field(..., description="Name of the tool to invoke")
        arguments: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments")
        thought: Optional[str] = Field(None, description="Optional reasoning before call")

    class RetrievalResultSchema(BaseModel):
        """Schema for a single retrieved document result."""
        content: str = Field(..., description="Document content")
        source: Optional[str] = Field(None, description="Document source URL or ID")
        score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Relevance score")
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class RankedAnswersSchema(BaseModel):
        """Multiple candidate answers with scores."""
        answers: List[AnswerSchema] = Field(..., min_length=1)
        best_index: int = Field(0, ge=0)

        @property
        def best(self) -> AnswerSchema:
            return self.answers[self.best_index]

else:
    # Stub dataclasses for environments without pydantic
    @dataclass
    class AnswerSchema:  # type: ignore[no-redef]
        answer: str
        sources: List[str] = field(default_factory=list)
        confidence: float = 1.0
        reasoning: Optional[str] = None

    @dataclass
    class ToolCallSchema:  # type: ignore[no-redef]
        tool_name: str
        arguments: Dict[str, Any] = field(default_factory=dict)
        thought: Optional[str] = None

    @dataclass
    class RetrievalResultSchema:  # type: ignore[no-redef]
        content: str
        source: Optional[str] = None
        score: Optional[float] = None
        metadata: Dict[str, Any] = field(default_factory=dict)

    @dataclass
    class RankedAnswersSchema:  # type: ignore[no-redef]
        answers: List[Any] = field(default_factory=list)
        best_index: int = 0

        @property
        def best(self) -> Any:
            return self.answers[self.best_index]


# ─────────────────────────────────────────────
# Field-level schema definition (for StructuredOutputParser)
# ─────────────────────────────────────────────

@dataclass
class FieldSchema:
    """Declares a single field expected in an LLM output."""
    name: str
    description: str
    dtype: str = "str"          # str | int | float | bool | list | dict
    required: bool = True
    aliases: List[str] = field(default_factory=list)
    default: Any = None
    choices: Optional[List[Any]] = None  # Enum-like constraint

    VALID_DTYPES = {"str", "int", "float", "bool", "list", "dict"}

    def __post_init__(self) -> None:
        if self.dtype not in self.VALID_DTYPES:
            raise ValueError(f"dtype {self.dtype!r} must be one of {self.VALID_DTYPES}")

    @property
    def all_names(self) -> List[str]:
        return [self.name] + self.aliases

    def validate_value(self, value: Any) -> ValidationResult:
        """Validate a parsed value against this field's constraints."""
        if value is None:
            if self.required:
                return ValidationResult(
                    status=ValidationStatus.FAILED,
                    field=self.name,
                    message=f"Required field '{self.name}' is missing",
                )
            return ValidationResult(status=ValidationStatus.SKIPPED, field=self.name)

        if self.choices and value not in self.choices:
            return ValidationResult(
                status=ValidationStatus.FAILED,
                field=self.name,
                message=f"Value {value!r} not in allowed choices {self.choices}",
                value=value,
            )
        return ValidationResult(status=ValidationStatus.PASSED, field=self.name, value=value)
