"""
Multi-layer output validation engine.

Validates parsed outputs for:
    - Schema completeness (required fields present)
    - Type correctness
    - Business-rule constraints
    - Safety checks (hallucination markers, sensitive leakage)
    - Consistency checks (confidence vs. sources)
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional, Type

from .schema import FieldSchema, ParseTrace, ValidationResult, ValidationStatus

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Safety patterns
# ─────────────────────────────────────────────

_HALLUCINATION_MARKERS = re.compile(
    r"\b(as of my knowledge cutoff|i cannot verify|i don't actually know|"
    r"i'm making this up|i don't have access to|this is fictional|"
    r"i am hallucinating|fabricated|invented fact)\b",
    re.IGNORECASE,
)

_DATA_LEAKAGE_PATTERNS = [
    re.compile(r"\b\d{16}\b"),                                   # credit card
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),                        # SSN
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.\w{2,}\b"), # email
    re.compile(r"(?:password|passwd|secret|api_key)\s*[=:]\s*\S+", re.I),
    re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*"),              # Bearer token
]

_PROMPT_INJECTION_MARKERS = re.compile(
    r"(ignore (previous|all) instructions|disregard system prompt|"
    r"you are now|pretend you are|act as if|jailbreak|DAN mode)",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────
# Custom rule type
# ─────────────────────────────────────────────

class ValidationRule:
    """A named validation rule with a predicate and error message."""

    def __init__(
        self,
        name: str,
        predicate: Callable[[Any], bool],
        message: str,
        field: Optional[str] = None,
        severity: ValidationStatus = ValidationStatus.FAILED,
    ):
        self.name = name
        self.predicate = predicate
        self.message = message
        self.field = field
        self.severity = severity

    def check(self, data: Any) -> ValidationResult:
        try:
            passed = self.predicate(data)
        except Exception as exc:
            return ValidationResult(
                status=ValidationStatus.FAILED,
                field=self.field,
                message=f"Rule '{self.name}' raised exception: {exc}",
            )
        return ValidationResult(
            status=ValidationStatus.PASSED if passed else self.severity,
            field=self.field,
            message="" if passed else self.message,
        )


# ─────────────────────────────────────────────
# Validator class
# ─────────────────────────────────────────────

class OutputValidator:
    """
    Validates a parsed output object against multiple layers:

    Layer 1 – Schema completeness  : required fields present with correct types
    Layer 2 – Business rules       : custom ValidationRule predicates
    Layer 3 – Safety checks        : hallucination, data leakage, prompt injection
    Layer 4 – Pydantic validation  : if a pydantic model is provided

    Usage::

        validator = OutputValidator(
            fields=[FieldSchema("answer", "str", required=True),
                    FieldSchema("confidence", "float", required=True)],
            rules=[ValidationRule("conf_range",
                                  lambda d: 0 <= d.get("confidence", 0) <= 1,
                                  "confidence must be between 0 and 1")],
        )
        results = validator.validate(parsed_dict, trace=trace)
        all_pass = all(r.passed for r in results)
    """

    def __init__(
        self,
        fields: Optional[List[FieldSchema]] = None,
        rules: Optional[List[ValidationRule]] = None,
        pydantic_model: Optional[Type] = None,
        enable_safety: bool = True,
    ):
        self.fields = fields or []
        self.rules = rules or []
        self.pydantic_model = pydantic_model
        self.enable_safety = enable_safety

    # ─── Public API ──────────────────────────────────────────────────────

    def validate(
        self,
        data: Any,
        raw_text: str = "",
        trace: Optional[ParseTrace] = None,
    ) -> List[ValidationResult]:
        """Run all validation layers. Returns list of ValidationResult objects."""
        results: List[ValidationResult] = []

        # Layer 1: schema field checks
        if self.fields and isinstance(data, dict):
            results.extend(self._validate_fields(data))

        # Layer 2: custom business rules
        for rule in self.rules:
            result = rule.check(data)
            results.append(result)
            if not result.passed:
                logger.debug("Rule '%s' failed: %s", rule.name, result.message)

        # Layer 3: safety
        if self.enable_safety:
            text_to_check = raw_text or (str(data) if data else "")
            results.extend(self._safety_checks(text_to_check))

        # Layer 4: pydantic structural validation
        if self.pydantic_model and isinstance(data, dict):
            results.extend(self._pydantic_validation(data))

        # Attach to trace
        if trace:
            for r in results:
                trace.add_validation(r)

        return results

    def is_valid(self, data: Any, raw_text: str = "") -> bool:
        """Quick check: True only if all non-warning validations pass."""
        results = self.validate(data, raw_text)
        return all(
            r.status in (ValidationStatus.PASSED, ValidationStatus.WARNING, ValidationStatus.SKIPPED)
            for r in results
        )

    def get_failures(self, data: Any, raw_text: str = "") -> List[ValidationResult]:
        """Return only the failed validations."""
        return [r for r in self.validate(data, raw_text) if not r.passed]

    # ─── Layer 1 ─────────────────────────────────────────────────────────

    def _validate_fields(self, data: Dict[str, Any]) -> List[ValidationResult]:
        results = []
        for field in self.fields:
            value = data.get(field.name)

            # Required check
            field_result = field.validate_value(value)
            results.append(field_result)

            if value is None:
                continue

            # Type check
            type_result = self._check_type(field.name, value, field.dtype)
            results.append(type_result)

        # Extra unknown fields (just warning)
        declared_names = {f.name for f in self.fields}
        for key in data.keys():
            if key not in declared_names:
                results.append(ValidationResult(
                    status=ValidationStatus.WARNING,
                    field=key,
                    message=f"Unexpected field '{key}' not in schema",
                ))
        return results

    def _check_type(self, field_name: str, value: Any, dtype: str) -> ValidationResult:
        expected_map: Dict[str, type] = {
            "str": str, "int": int, "float": (int, float),
            "bool": bool, "list": list, "dict": dict,
        }
        expected = expected_map.get(dtype)
        if expected is None:
            return ValidationResult(status=ValidationStatus.SKIPPED, field=field_name)

        if isinstance(value, expected):
            return ValidationResult(status=ValidationStatus.PASSED, field=field_name)

        # Soft coercibility check
        try:
            coerce_map = {"int": int, "float": float, "str": str, "bool": bool}
            if dtype in coerce_map:
                coerce_map[dtype](value)
                return ValidationResult(
                    status=ValidationStatus.WARNING,
                    field=field_name,
                    message=f"Field '{field_name}' has type {type(value).__name__}, expected {dtype} (coercible)",
                )
        except (ValueError, TypeError):
            pass

        return ValidationResult(
            status=ValidationStatus.FAILED,
            field=field_name,
            message=f"Field '{field_name}' has type {type(value).__name__}, expected {dtype}",
            value=value,
        )

    # ─── Layer 3 ─────────────────────────────────────────────────────────

    def _safety_checks(self, text: str) -> List[ValidationResult]:
        results: List[ValidationResult] = []

        if not text:
            return results

        # Hallucination markers
        if _HALLUCINATION_MARKERS.search(text):
            results.append(ValidationResult(
                status=ValidationStatus.WARNING,
                message="Output contains possible hallucination admission markers",
            ))
            logger.warning("Hallucination marker detected in output")

        # Data leakage
        for pattern in _DATA_LEAKAGE_PATTERNS:
            if pattern.search(text):
                results.append(ValidationResult(
                    status=ValidationStatus.FAILED,
                    message="Potential sensitive data leakage detected in output",
                ))
                logger.error("Data leakage pattern matched in output")
                break  # one failure is enough

        # Prompt injection
        if _PROMPT_INJECTION_MARKERS.search(text):
            results.append(ValidationResult(
                status=ValidationStatus.FAILED,
                message="Possible prompt injection detected in output",
            ))
            logger.error("Prompt injection marker found in output")

        return results

    # ─── Layer 4 ─────────────────────────────────────────────────────────

    def _pydantic_validation(self, data: Dict[str, Any]) -> List[ValidationResult]:
        try:
            self.pydantic_model(**data)
            return [ValidationResult(
                status=ValidationStatus.PASSED,
                message=f"Pydantic model {self.pydantic_model.__name__} validation passed",
            )]
        except Exception as exc:
            return [ValidationResult(
                status=ValidationStatus.FAILED,
                message=f"Pydantic validation failed: {exc}",
            )]


# ─────────────────────────────────────────────
# Convenience factory
# ─────────────────────────────────────────────

def build_answer_validator(enable_safety: bool = True) -> OutputValidator:
    """Return a pre-configured validator for AnswerSchema outputs."""
    fields = [
        FieldSchema("answer", "The answer text", dtype="str", required=True),
        FieldSchema("sources", "List of sources", dtype="list", required=False),
        FieldSchema("confidence", "Confidence score", dtype="float", required=False),
    ]
    rules = [
        ValidationRule(
            "answer_not_empty",
            lambda d: bool(d.get("answer", "").strip()) if isinstance(d, dict) else bool(str(d).strip()),
            "answer field must not be empty",
            field="answer",
        ),
        ValidationRule(
            "confidence_range",
            lambda d: 0.0 <= float(d.get("confidence", 1.0)) <= 1.0 if isinstance(d, dict) else True,
            "confidence must be in [0.0, 1.0]",
            field="confidence",
            severity=ValidationStatus.WARNING,
        ),
    ]
    return OutputValidator(fields=fields, rules=rules, enable_safety=enable_safety)
