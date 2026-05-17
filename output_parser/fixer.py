"""
Fault-tolerant output repair engine.

When an LLM returns malformed, incomplete, or broken output, OutputFixer
attempts progressively deeper repair strategies before giving up:

    Strategy 1  REGEX_REPAIR    — Lightweight regex-based JSON/text fixes
    Strategy 2  FIELD_INJECTION — Inject missing required fields with defaults
    Strategy 3  FALLBACK_PARSE  — Try alternative parsers (YAML, key-value, …)
    Strategy 4  LLM_REFORMAT    — Ask the LLM to reformat its own output (if llm_fn provided)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from .schema import  FixStrategy, OutputFormat

logger = logging.getLogger(__name__)

# Type alias for an LLM callable used for reformat requests
LLMCallable = Callable[[str], str]


# ─────────────────────────────────────────────
# Repair utilities (stateless helpers)
# ─────────────────────────────────────────────

def _strip_markdown_fences(text: str) -> str:
    """Remove ```lang ... ``` fences and return inner content."""
    match = re.search(r"```(?:\w+)?\s*\n?([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _fix_json_quotes(text: str) -> str:
    """Convert single-quoted JSON to double-quoted."""
    # Replace unescaped single quotes used as JSON delimiters
    result = re.sub(r"(?<![\\])'", '"', text)
    return result


def _fix_trailing_commas(text: str) -> str:
    """Remove trailing commas before } or ]."""
    return re.sub(r",\s*([}\]])", r"\1", text)


def _fix_unquoted_keys(text: str) -> str:
    """Quote bare keys in JSON-like structures: {key: value} → {"key": value}."""
    return re.sub(r'([{,]\s*)([a-zA-Z_]\w*)\s*:', r'\1"\2":', text)


def _truncate_to_valid_json(text: str) -> Optional[str]:
    """
    Try to find a valid JSON object by progressively trimming from the end.
    Handles LLM outputs that start valid JSON then trail off.
    """
    # Find last brace/bracket that closes the first opener
    opener = None
    closer = None
    for ch in text:
        if ch in "{[":
            opener, closer = ch, ("}" if ch == "{" else "]")
            break
    if not opener:
        return None

    # Try trimming from the right
    candidate = text[text.find(opener):]
    for i in range(len(candidate), 0, -1):
        try:
            json.loads(candidate[:i])
            return candidate[:i]
        except json.JSONDecodeError:
            continue
    return None


def _extract_json_block(text: str) -> Optional[str]:
    """Find and return the first complete JSON block in text."""
    decoder = json.JSONDecoder()
    for start_char in ("{", "["):
        idx = text.find(start_char)
        while idx != -1:
            try:
                _, end = decoder.raw_decode(text, idx)
                return text[idx:end]
            except json.JSONDecodeError:
                idx = text.find(start_char, idx + 1)
    return None


def _parse_key_value_fallback(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract key: value pairs from plain text as a last resort.
    Returns a dict or None.
    """
    result: Dict[str, Any] = {}
    pattern = re.compile(
        r"(?m)^\s*\*?\*?(?P<key>[A-Za-z_][\w\s]*?)\*?\*?\s*[:=]\s*(?P<val>.+?)$"
    )
    for m in pattern.finditer(text):
        key = m.group("key").strip().lower().replace(" ", "_")
        val = m.group("val").strip().strip("'\"")
        if key and val:
            result[key] = val
    return result if result else None


# ─────────────────────────────────────────────
# Main Fixer class
# ─────────────────────────────────────────────

class OutputFixer:
    """
    Attempts to repair malformed LLM outputs using a hierarchy of strategies.

    Usage::

        fixer = OutputFixer(required_fields=["answer", "sources"])
        fixed_text, strategy = fixer.fix(raw_broken_text, expected_format=OutputFormat.JSON)
        # → ("{ \"answer\": ... }", FixStrategy.REGEX_REPAIR)
    """

    _JSON_REPAIR_PIPELINE: List[Callable[[str], str]] = [
        _strip_markdown_fences,
        _fix_trailing_commas,
        _fix_json_quotes,
        _fix_unquoted_keys,
    ]

    def __init__(
        self,
        required_fields: Optional[List[str]] = None,
        field_defaults: Optional[Dict[str, Any]] = None,
        llm_fn: Optional[LLMCallable] = None,
    ):
        """
        Args:
            required_fields: Field names that MUST be present in the output dict.
            field_defaults:  Default values injected when a required field is missing.
            llm_fn:          Optional callable that sends a prompt to an LLM and returns
                             its response string (used for LLM_REFORMAT strategy).
        """
        self.required_fields = required_fields or []
        self.field_defaults = field_defaults or {}
        self.llm_fn = llm_fn

    # ─── Public API ──────────────────────────────────────────────────────

    def fix(
        self,
        text: str,
        expected_format: OutputFormat = OutputFormat.JSON,
    ) -> Tuple[str, FixStrategy]:
        """
        Attempt to fix malformed text.

        Returns:
            (repaired_text, strategy_used)  — caller should re-parse the repaired text.
        """
        if not text or not text.strip():
            return text, FixStrategy.NONE

        logger.info("OutputFixer: attempting repair for format=%s", expected_format.value)

        # Strategy 1: Regex repair
        repaired, strategy = self._try_regex_repair(text, expected_format)
        if strategy != FixStrategy.NONE:
            logger.info("Regex repair succeeded")
            return repaired, strategy

        # Strategy 2: Field injection (dict → inject defaults)
        repaired, strategy = self._try_field_injection(text)
        if strategy != FixStrategy.NONE:
            logger.info("Field injection succeeded")
            return repaired, strategy

        # Strategy 3: Fallback parse (key-value, then re-serialize as JSON)
        repaired, strategy = self._try_fallback_parse(text)
        if strategy != FixStrategy.NONE:
            logger.info("Fallback parse succeeded")
            return repaired, strategy

        # Strategy 4: LLM reformat
        if self.llm_fn:
            repaired, strategy = self._try_llm_reformat(text, expected_format)
            if strategy != FixStrategy.NONE:
                logger.info("LLM reformat succeeded")
                return repaired, strategy

        logger.warning("OutputFixer: all strategies exhausted — returning original text")
        return text, FixStrategy.NONE

    def fix_dict(
        self,
        data: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], FixStrategy]:
        """
        Fix a partially parsed dict by injecting missing required fields.
        Returns (fixed_dict, strategy).
        """
        if not self.required_fields:
            return data, FixStrategy.NONE

        missing = [f for f in self.required_fields if f not in data or data[f] is None]
        if not missing:
            return data, FixStrategy.NONE

        fixed = dict(data)
        for f in missing:
            fixed[f] = self.field_defaults.get(f, None)
            logger.debug("Injected default for missing field '%s'", f)

        return fixed, FixStrategy.FIELD_INJECTION

    # ─── Private strategies ───────────────────────────────────────────────

    def _try_regex_repair(
        self, text: str, expected_format: OutputFormat
    ) -> Tuple[str, FixStrategy]:
        """Apply regex-based repairs for JSON and text formats."""

        if expected_format == OutputFormat.JSON:
            candidate = text

            # Step through repair pipeline
            for fn in self._JSON_REPAIR_PIPELINE:
                candidate = fn(candidate)

            # Try to parse
            try:
                json.loads(candidate)
                return candidate, FixStrategy.REGEX_REPAIR
            except json.JSONDecodeError:
                pass

            # Try extracting a JSON block
            extracted = _extract_json_block(candidate)
            if extracted:
                try:
                    json.loads(extracted)
                    return extracted, FixStrategy.REGEX_REPAIR
                except json.JSONDecodeError:
                    pass

            # Try truncation to valid JSON
            truncated = _truncate_to_valid_json(candidate)
            if truncated:
                return truncated, FixStrategy.REGEX_REPAIR

        # For non-JSON: just strip markdown fences
        stripped = _strip_markdown_fences(text)
        if stripped != text and stripped:
            return stripped, FixStrategy.REGEX_REPAIR

        return text, FixStrategy.NONE

    def _try_field_injection(self, text: str) -> Tuple[str, FixStrategy]:
        """Parse what we can, inject missing fields, re-serialize."""
        if not self.required_fields:
            return text, FixStrategy.NONE

        # Try JSON first
        try:
            data = json.loads(_extract_json_block(text) or text)
            if isinstance(data, dict):
                fixed, strategy = self.fix_dict(data)
                if strategy != FixStrategy.NONE:
                    return json.dumps(fixed, ensure_ascii=False), FixStrategy.FIELD_INJECTION
        except (json.JSONDecodeError, TypeError):
            pass

        return text, FixStrategy.NONE

    def _try_fallback_parse(self, text: str) -> Tuple[str, FixStrategy]:
        """Parse key-value text and re-serialize as JSON."""
        kv = _parse_key_value_fallback(text)
        if kv:
            # Inject defaults for missing fields
            for f in self.required_fields:
                if f not in kv:
                    kv[f] = self.field_defaults.get(f, None)
            try:
                return json.dumps(kv, ensure_ascii=False), FixStrategy.FALLBACK_PARSE
            except (TypeError, ValueError):
                pass
        return text, FixStrategy.NONE

    def _try_llm_reformat(
        self, text: str, expected_format: OutputFormat
    ) -> Tuple[str, FixStrategy]:
        """Send a reformat request to the LLM."""
        assert self.llm_fn is not None

        fields_hint = ""
        if self.required_fields:
            fields_hint = f"Required fields: {', '.join(self.required_fields)}.\n"

        format_hint = {
            OutputFormat.JSON: "valid JSON object",
            OutputFormat.YAML: "valid YAML",
            OutputFormat.CSV: "CSV with header row",
        }.get(expected_format, "structured text")

        prompt = (
            f"The following text is a malformed LLM output that could not be parsed.\n"
            f"Please reformat it as {format_hint}.\n"
            f"{fields_hint}"
            f"Return ONLY the reformatted content with no extra explanation.\n\n"
            f"--- BEGIN MALFORMED OUTPUT ---\n{text}\n--- END ---"
        )

        try:
            reformatted = self.llm_fn(prompt)
            if reformatted and reformatted.strip():
                return reformatted.strip(), FixStrategy.LLM_REFORMAT
        except Exception as exc:
            logger.warning("LLM reformat call failed: %s", exc)

        return text, FixStrategy.NONE


# ─────────────────────────────────────────────
# Convenience factory
# ─────────────────────────────────────────────

def build_answer_fixer(llm_fn: Optional[LLMCallable] = None) -> OutputFixer:
    """Return a pre-configured OutputFixer for AnswerSchema outputs."""
    return OutputFixer(
        required_fields=["answer"],
        field_defaults={"answer": "", "sources": [], "confidence": 0.5},
        llm_fn=llm_fn,
    )
