"""
parser.py — AI-Powered Output Parsing & Validation Engine (Production-Grade).

The main entry point for the RAG Output Parser module. Orchestrates:
    FormatDetector → Format-specific Parsers → OutputFixer → OutputValidator
    → RetryHandler → Typed Output → ParseResult with full audit trace.

Supports:
    • JSON, YAML, CSV, Markdown Table, Numbered/Bulleted List, Key-Value, XML,
      Tool Call, Plain Text, Mixed outputs
    • Strict, Lenient, and Semantic parse modes
    • Schema enforcement via Pydantic or FieldSchema definitions
    • Fault tolerance: regex repair → field injection → LLM reformat
    • Retry & regeneration with escalating prompt strategies
    • Safety checks: hallucination, data leakage, prompt injection
    • Observability: full ParseTrace, structured logging
    • Caching of successful parse results
    • Typed output via Pydantic models or dataclasses
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import re
import time
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

try:
    import yaml
    _YAML_OK = True
except ImportError:
    _YAML_OK = False

from .fixer import OutputFixer, build_answer_fixer
from .format_detector import FormatDetector
from .retry_handler import RetryHandler, graceful_fallback as retry_graceful_fallback
from .schema import (
    AnswerSchema,
    FieldSchema,
    FixStrategy,
    OutputFormat,
    ParseMode,
    ParseResult,
    ParseTrace,
    ValidationStatus,
)
from .validator import OutputValidator, build_answer_validator

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ─────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────

class ParseError(Exception):
    """Raised when the parser cannot produce a valid output."""
    def __init__(self, message: str, trace: Optional[ParseTrace] = None):
        super().__init__(message)
        self.trace = trace


# ─────────────────────────────────────────────
# Per-format parse helpers (fast, no class overhead)
# ─────────────────────────────────────────────

def _parse_json(text: str) -> Any:
    """Extract and parse the first valid JSON block from text."""
    # Try code-fence first
    fence = re.search(r"```(?:json)?\s*\n?([\s\S]*?)```", text, re.I)
    if fence:
        candidate = fence.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # raw_decode scan
    decoder = json.JSONDecoder()
    for start in ("{", "["):
        idx = text.find(start)
        while idx != -1:
            try:
                obj, _ = decoder.raw_decode(text, idx)
                return obj
            except json.JSONDecodeError:
                idx = text.find(start, idx + 1)

    raise ValueError("No valid JSON found")


def _parse_yaml(text: str) -> Any:
    if not _YAML_OK:
        raise ImportError("pyyaml not installed")
    fence = re.search(r"```(?:yaml|yml)?\s*\n?([\s\S]*?)```", text, re.I)
    raw = fence.group(1).strip() if fence else text.strip()
    result = yaml.safe_load(raw)
    if result is None:
        raise ValueError("YAML parsed to None")
    return result


def _parse_csv(text: str) -> List[Dict[str, str]]:
    fence = re.search(r"```(?:csv)?\s*\n?([\s\S]*?)```", text, re.I)
    raw = fence.group(1).strip() if fence else text.strip()
    reader = csv.DictReader(io.StringIO(raw))
    rows = list(reader)
    if not rows:
        raise ValueError("CSV has no data rows")
    return [{k.strip(): v.strip() for k, v in r.items() if k} for r in rows]


def _parse_markdown_table(text: str) -> List[Dict[str, str]]:
    lines = [l for l in text.splitlines() if l.strip()]
    table_lines = [l for l in lines if "|" in l]
    if len(table_lines) < 2:
        raise ValueError("Not enough table rows")
    header = [h.strip() for h in table_lines[0].split("|") if h.strip()]
    data_lines = [l for l in table_lines[1:] if not re.match(r"^\s*\|?[-| :]+\|?\s*$", l)]
    result = []
    for line in data_lines:
        cells = [c.strip() for c in line.split("|") if c.strip() or True]
        cells = [c for c in cells if c != ""]  # drop empty boundary cells
        if len(cells) >= len(header):
            result.append(dict(zip(header, cells)))
    return result


def _parse_list(text: str) -> List[str]:
    numbered = re.compile(r"^\s*(?:\d+[\.\)]|[a-z][\.\)])\s+(.+)$", re.M)
    bulleted = re.compile(r"^\s*[*\-–•▸]\s+(.+)$", re.M)
    items = numbered.findall(text) or bulleted.findall(text)
    if not items:
        # Comma fallback
        single = re.sub(r"\s+", " ", text.strip())
        items = [i.strip() for i in single.split(",") if i.strip()]
    if not items:
        raise ValueError("No list items found")
    return [i.strip() for i in items]


def _parse_xml(text: str) -> Dict[str, str]:
    pattern = re.compile(r"<(?P<tag>[a-zA-Z][\w\-]*)>(?P<val>.*?)</(?P=tag)>", re.DOTALL)
    result = {}
    for m in pattern.finditer(text):
        result[m.group("tag")] = m.group("val").strip()
    if not result:
        raise ValueError("No XML tag pairs found")
    return result


def _parse_tool_call(text: str) -> Dict[str, Any]:
    """Parse ReAct-style or OpenAI function-call tool invocations."""
    # OpenAI function call JSON
    try:
        data = _parse_json(text)
        if isinstance(data, dict) and "name" in data and "arguments" in data:
            args = data["arguments"]
            if isinstance(args, str):
                args = json.loads(args)
            return {"tool_name": data["name"], "arguments": args, "thought": data.get("thought")}
    except (ValueError, json.JSONDecodeError):
        pass

    # ReAct-style: Action: tool_name\nAction Input: {...}
    action_match = re.search(
        r"(?:Action|Tool|Function|tool_name)\s*[:\-]\s*(?P<name>[\w\.]+)", text, re.I
    )
    if not action_match:
        raise ValueError("No tool call pattern found")

    tool_name = action_match.group("name").strip()
    arg_match = re.search(
        r"(?:Action Input|Arguments?|Input|Args?)\s*[:\-]\s*(?P<args>\{[\s\S]+?\}|\S+)",
        text, re.I,
    )
    arguments: Any = {}
    if arg_match:
        raw_args = arg_match.group("args").strip()
        try:
            arguments = json.loads(raw_args)
        except (json.JSONDecodeError, ValueError):
            arguments = {"input": raw_args}

    thought_match = re.search(
        r"(?:Thought|Reasoning)\s*[:\-]\s*(?P<t>.+?)(?=\n(?:Action|Tool)|$)", text, re.I | re.DOTALL
    )
    thought = thought_match.group("t").strip() if thought_match else None
    return {"tool_name": tool_name, "arguments": arguments, "thought": thought}


def _parse_key_value(text: str) -> Dict[str, str]:
    pattern = re.compile(
        r"(?m)^\s*\*?\*?(?P<key>[A-Za-z_][\w\s]*?)\*?\*?\s*[:=]\s*(?P<val>.+?)$"
    )
    result = {}
    for m in pattern.finditer(text):
        key = m.group("key").strip().lower().replace(" ", "_")
        val = m.group("val").strip()
        if key and val:
            result[key] = val
    if not result:
        raise ValueError("No key: value pairs found")
    return result


def _parse_plain_text(text: str) -> Dict[str, str]:
    return {"answer": text.strip()}


# Format → parse function mapping
_FORMAT_PARSERS: Dict[OutputFormat, Callable[[str], Any]] = {
    OutputFormat.JSON:           _parse_json,
    OutputFormat.YAML:           _parse_yaml,
    OutputFormat.CSV:            _parse_csv,
    OutputFormat.MARKDOWN_TABLE: _parse_markdown_table,
    OutputFormat.NUMBERED_LIST:  _parse_list,
    OutputFormat.BULLETED_LIST:  _parse_list,
    OutputFormat.XML:            _parse_xml,
    OutputFormat.TOOL_CALL:      _parse_tool_call,
    OutputFormat.KEY_VALUE:      _parse_key_value,
    OutputFormat.PLAIN_TEXT:     _parse_plain_text,
    OutputFormat.MIXED:          _parse_plain_text,
    OutputFormat.UNKNOWN:        _parse_plain_text,
}


# ─────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────

def _cache_key(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()


# ─────────────────────────────────────────────
# Main Parser class
# ─────────────────────────────────────────────

class OutputParser:
    """
    AI-Powered Output Parsing & Validation Engine.

    Converts raw LLM text into validated, typed Python objects using a
    multi-stage pipeline:

        Detect → Parse → Fix → Validate → Retry → Typed Output

    Quick start::

        parser = OutputParser()
        result = parser.parse('{"answer": "Paris", "confidence": 0.98}')
        print(result.data)   # {'answer': 'Paris', 'confidence': 0.98}

    Typed output::

        from rag.output_parser.schema import AnswerSchema
        parser = OutputParser(schema=AnswerSchema, mode=ParseMode.STRICT)
        result = parser.parse(raw_text)
        answer: AnswerSchema = result.data

    With LLM retry::

        parser = OutputParser(
            schema=AnswerSchema,
            llm_fn=my_llm,
            max_retries=3,
        )
    """

    def __init__(
        self,
        schema: Optional[Type[T]] = None,
        fields: Optional[List[FieldSchema]] = None,
        mode: ParseMode = ParseMode.LENIENT,
        expected_format: Optional[OutputFormat] = None,
        llm_fn: Optional[Callable[[str], str]] = None,
        max_retries: int = 2,
        enable_safety: bool = True,
        enable_cache: bool = True,
        original_prompt: str = "",
    ):
        """
        Args:
            schema:           Optional Pydantic model class. Parsed dicts are
                              cast to this type if provided.
            fields:           Explicit FieldSchema list (alternative to pydantic schema).
            mode:             ParseMode.STRICT | LENIENT | SEMANTIC | TOOL_CALL
            expected_format:  Force a specific OutputFormat (skip detection).
            llm_fn:           LLM callable for fix/retry operations.
            max_retries:      Maximum regeneration retries on parse failure.
            enable_safety:    Run safety validation checks (default True).
            enable_cache:     Cache successful parse results (default True).
            original_prompt:  The original user prompt (used in retry prompts).
        """
        self.schema = schema
        self.fields = fields or []
        self.mode = mode
        self.expected_format = expected_format
        self.llm_fn = llm_fn
        self.max_retries = max_retries
        self.enable_safety = enable_safety
        self.enable_cache = enable_cache
        self.original_prompt = original_prompt

        # Sub-components
        self._detector = FormatDetector()
        self._validator = self._build_validator()
        self._fixer = self._build_fixer()
        self._retry: Optional[RetryHandler] = (
            RetryHandler(
                llm_fn=llm_fn,
                max_retries=max_retries,
                pydantic_schema=schema,
                required_fields=[f.name for f in self.fields] if self.fields else [],
            )
            if llm_fn and max_retries > 0
            else None
        )
        self._cache: Dict[str, ParseResult] = {}

    # ─── Public API ──────────────────────────────────────────────────────

    def parse(
        self,
        text: str,
        expected_format: Optional[OutputFormat] = None,
    ) -> ParseResult:
        """
        Parse raw LLM text into a validated, optionally typed ParseResult.

        Args:
            text:            Raw LLM output string.
            expected_format: Override the detected format for this call only.

        Returns:
            ParseResult with .data (typed or dict), .trace (audit trail), .ok

        Raises:
            ParseError: In STRICT mode if parsing fails even after fix/retry.
        """
        if not text or not text.strip():
            trace = ParseTrace(raw_input=text, success=False)
            trace.add_error("Input text is empty")
            return ParseResult(data=None, trace=trace, raw=text)

        # Cache hit
        cache_key = _cache_key(text)
        if self.enable_cache and cache_key in self._cache:
            logger.debug("Cache hit for output parse")
            return self._cache[cache_key]

        trace = ParseTrace(raw_input=text, parse_mode=self.mode)
        t0 = time.perf_counter()

        result = self._run_pipeline(text, expected_format, trace)

        trace.duration_ms = (time.perf_counter() - t0) * 1000

        if result.ok and self.enable_cache:
            self._cache[cache_key] = result

        return result

    def parse_typed(
        self, text: str, schema: Type[T], expected_format: Optional[OutputFormat] = None
    ) -> T:
        """
        Parse and return a typed instance of `schema`.
        Equivalent to parser.parse(text).as_typed(schema).
        """
        result = self.parse(text, expected_format)
        if not result.ok:
            raise ParseError("Parsing produced no data", trace=result.trace)
        return result.as_typed(schema)

    def get_format_instructions(self) -> str:
        """Return LLM format instructions based on schema/mode config."""
        if self.schema:
            try:
                schema_json = json.dumps(
                    self.schema.model_json_schema(), indent=2, ensure_ascii=False
                )
                return (
                    f"Return ONLY valid JSON matching this schema:\n"
                    f"```json\n{schema_json}\n```\n"
                    f"Do not include any explanation outside the JSON block."
                )
            except AttributeError:
                pass
        if self.fields:
            lines = ["Return output using the following key: value format:"]
            for f in self.fields:
                req = "(required)" if f.required else "(optional)"
                lines.append(f"  {f.name}: <{f.dtype}> — {f.description} {req}")
            return "\n".join(lines)
        return (
            "Return ONLY valid JSON. Example:\n"
            '{"answer": "...", "sources": [], "confidence": 0.9}'
        )

    def clear_cache(self) -> None:
        self._cache.clear()
        logger.debug("Parse cache cleared")

    # ─── Pipeline ────────────────────────────────────────────────────────

    def _run_pipeline(
        self,
        text: str,
        override_format: Optional[OutputFormat],
        trace: ParseTrace,
    ) -> ParseResult:

        # 1. Detect format
        fmt = override_format or self.expected_format or self._detector.detect(text)
        trace.detected_format = fmt
        logger.debug("Detected format: %s", fmt.value)

        # 2. Parse
        data, parse_ok = self._attempt_parse(text, fmt, trace)

        # 3. Fix if parse failed
        if not parse_ok:
            data, parse_ok = self._attempt_fix_and_parse(text, fmt, trace)

        # 4. Retry with LLM if still failing
        if not parse_ok and self._retry:
            data, parse_ok = self._attempt_retry(text, fmt, trace)

        # 5. Graceful fallback
        if not parse_ok:
            if self.mode == ParseMode.STRICT:
                trace.success = False
                raise ParseError(
                    "Failed to parse output in STRICT mode after all recovery attempts",
                    trace=trace,
                )
            data = retry_graceful_fallback(text, self.original_prompt)
            trace.add_warning("Using graceful fallback — parse failed after all recovery")
            parse_ok = True

        # 6. Cast to typed schema
        if data and self.schema:
            data = self._cast_to_schema(data, trace)

        # 7. Validate
        if data is not None:
            self._validate(data, text, trace)

        trace.success = parse_ok and data is not None
        return ParseResult(
            data=data,
            schema_type=self.schema,
            trace=trace,
            raw=text,
        )

    # ─── Parse attempt ────────────────────────────────────────────────────

    def _attempt_parse(
        self, text: str, fmt: OutputFormat, trace: ParseTrace
    ) -> tuple:
        parse_fn = _FORMAT_PARSERS.get(fmt, _parse_plain_text)

        # Semantic mode: try LLM-assisted extraction first
        if self.mode == ParseMode.SEMANTIC and self.llm_fn:
            try:
                data = self._semantic_parse(text)
                logger.debug("Semantic parse succeeded")
                return data, True
            except Exception as exc:
                logger.debug("Semantic parse failed: %s — falling back to structural", exc)
                trace.add_warning(f"Semantic parse failed: {exc}")

        try:
            data = parse_fn(text)
            return data, True
        except Exception as exc:
            trace.add_error(f"Parse failed ({fmt.value}): {exc}")
            logger.debug("Parse failed: %s", exc)
            return None, False

    def _attempt_fix_and_parse(
        self, text: str, fmt: OutputFormat, trace: ParseTrace
    ) -> tuple:
        try:
            fixed_text, strategy = self._fixer.fix(text, expected_format=fmt)
            trace.fix_applied = strategy
            if strategy == FixStrategy.NONE:
                return None, False
            logger.info("Fixer applied strategy: %s", strategy.value)

            parse_fn = _FORMAT_PARSERS.get(fmt, _parse_plain_text)
            data = parse_fn(fixed_text)
            return data, True
        except Exception as exc:
            trace.add_error(f"Fix+parse failed: {exc}")
            return None, False

    def _attempt_retry(
        self, text: str, fmt: OutputFormat, trace: ParseTrace
    ) -> tuple:
        assert self._retry is not None
        parse_fn = _FORMAT_PARSERS.get(fmt, _parse_plain_text)
        last_error = trace.errors[-1] if trace.errors else ""

        retry_result = self._retry.run(
            original_prompt=self.original_prompt,
            parse_fn=parse_fn,
            format_instructions=self.get_format_instructions(),
            last_error=last_error,
            last_response=text,
        )
        trace.retries = retry_result.attempts

        if retry_result.success:
            try:
                data = parse_fn(retry_result.response)
                return data, True
            except Exception as exc:
                trace.add_error(f"Post-retry parse failed: {exc}")

        return None, False

    # ─── Semantic parsing ─────────────────────────────────────────────────

    def _semantic_parse(self, text: str) -> Any:
        """Use LLM to extract structured data from freeform text."""
        assert self.llm_fn is not None
        instructions = self.get_format_instructions()
        prompt = (
            f"Extract the key information from the following text and return it "
            f"as structured JSON.\n\n"
            f"{instructions}\n\n"
            f"Text to extract from:\n{text}\n\n"
            f"Return ONLY the JSON, nothing else:"
        )
        response = self.llm_fn(prompt)
        return _parse_json(response)

    # ─── Schema casting ───────────────────────────────────────────────────

    def _cast_to_schema(self, data: Any, trace: ParseTrace) -> Any:
        if isinstance(data, self.schema):
            return data
        try:
            if isinstance(data, dict):
                return self.schema(**data)
            if isinstance(data, str):
                return self.schema(answer=data)  # simple fallback for text
        except Exception as exc:
            trace.add_warning(f"Schema cast failed: {exc} — returning raw dict")
        return data

    # ─── Validation ───────────────────────────────────────────────────────

    def _validate(self, data: Any, raw_text: str, trace: ParseTrace) -> None:
        raw_data = data if isinstance(data, dict) else (data.__dict__ if hasattr(data, "__dict__") else {})
        results = self._validator.validate(raw_data, raw_text=raw_text, trace=trace)
        failed = [r for r in results if r.status == ValidationStatus.FAILED]
        warnings = [r for r in results if r.status == ValidationStatus.WARNING]
        if failed:
            msgs = "; ".join(r.message for r in failed)
            logger.warning("Validation failures: %s", msgs)
        if warnings:
            msgs = "; ".join(r.message for r in warnings)
            logger.info("Validation warnings: %s", msgs)

    # ─── Component factories ──────────────────────────────────────────────

    def _build_validator(self) -> OutputValidator:
        pydantic_model = self.schema if self.schema else None
        return OutputValidator(
            fields=self.fields,
            pydantic_model=pydantic_model,
            enable_safety=self.enable_safety,
        )

    def _build_fixer(self) -> OutputFixer:
        required = [f.name for f in self.fields if f.required]
        defaults = {f.name: f.default for f in self.fields}
        return OutputFixer(
            required_fields=required,
            field_defaults=defaults,
            llm_fn=self.llm_fn,
        )


# ─────────────────────────────────────────────
# Convenience factory functions
# ─────────────────────────────────────────────

def create_answer_parser(
    llm_fn: Optional[Callable[[str], str]] = None,
    max_retries: int = 2,
    strict: bool = False,
) -> OutputParser:
    """
    Create a pre-configured parser for standard RAG AnswerSchema outputs.

    Usage::
        parser = create_answer_parser(llm_fn=my_llm)
        result = parser.parse(raw_llm_output)
        answer: AnswerSchema = result.data
    """
    try:
        from pydantic import BaseModel
        use_pydantic = True
    except ImportError:
        use_pydantic = False

    return OutputParser(
        schema=AnswerSchema if use_pydantic else None,
        fields=[] if use_pydantic else [
            FieldSchema("answer", "The answer text", dtype="str", required=True),
            FieldSchema("sources", "Source list", dtype="list", required=False, default=[]),
            FieldSchema("confidence", "Confidence 0-1", dtype="float", required=False, default=1.0),
        ],
        mode=ParseMode.STRICT if strict else ParseMode.LENIENT,
        llm_fn=llm_fn,
        max_retries=max_retries,
        enable_safety=True,
    )


def create_tool_call_parser(
    llm_fn: Optional[Callable[[str], str]] = None,
) -> OutputParser:
    """Create a pre-configured parser for tool/function call outputs."""
    return OutputParser(
        mode=ParseMode.TOOL_CALL,
        expected_format=OutputFormat.TOOL_CALL,
        llm_fn=llm_fn,
        max_retries=1,
        enable_safety=False,
    )


def create_json_parser(
    schema: Optional[Type] = None,
    llm_fn: Optional[Callable[[str], str]] = None,
    strict: bool = False,
) -> OutputParser:
    """Create a pre-configured JSON parser with optional Pydantic schema."""
    return OutputParser(
        schema=schema,
        expected_format=OutputFormat.JSON,
        mode=ParseMode.STRICT if strict else ParseMode.LENIENT,
        llm_fn=llm_fn,
        max_retries=2,
    )
