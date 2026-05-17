"""
Multi-signal format detection for LLM outputs.

Analyses text using structural heuristics to determine its most likely format
before parsing begins. Returns a ranked list of candidates with confidence scores.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .schema import OutputFormat


# ─────────────────────────────────────────────
# Detection result
# ─────────────────────────────────────────────

@dataclass
class FormatCandidate:
    """A detected format with an associated confidence score."""
    format: OutputFormat
    confidence: float       # 0.0 – 1.0
    evidence: str = ""      # Human-readable reason

    def __repr__(self) -> str:
        return (
            f"FormatCandidate({self.format.value}, "
            f"confidence={self.confidence:.2f}, evidence={self.evidence!r})"
        )


# ─────────────────────────────────────────────
# Detector
# ─────────────────────────────────────────────

class FormatDetector:
    """
    Detects the format of raw LLM output using a multi-signal scoring system.

    Each detector method assigns a confidence score to one format. The detector
    returns ALL candidates above a minimum threshold, sorted by confidence.

    Usage::

        detector = FormatDetector()
        best = detector.detect(text)        # → OutputFormat
        all_candidates = detector.rank(text)  # → List[FormatCandidate]
    """

    _MIN_CONFIDENCE = 0.15  # Below this score, candidates are discarded

    # ─── Pre-compiled patterns ───────────────────────────────────────────
    _JSON_OBJECT_RE   = re.compile(r"^\s*\{[\s\S]*\}\s*$", re.MULTILINE)
    _JSON_ARRAY_RE    = re.compile(r"^\s*\[[\s\S]*\]\s*$", re.MULTILINE)
    _YAML_KEY_RE      = re.compile(r"^[a-zA-Z_][\w\-]*\s*:\s*\S", re.MULTILINE)
    _YAML_LIST_RE     = re.compile(r"^\s*-\s+\S", re.MULTILINE)
    _CSV_HEADER_RE    = re.compile(r"^[^\n,]+(?:,[^\n,]+){2,}\n", re.MULTILINE)
    _MD_TABLE_RE      = re.compile(r"^\|.+\|\s*\n\s*\|[-| :]+\|", re.MULTILINE)
    _NUMBERED_RE      = re.compile(r"^\s*(?:\d+[\.\)]|[a-z][\.\)])\s+\S", re.MULTILINE)
    _BULLETED_RE      = re.compile(r"^\s*[*\-–•]\s+\S", re.MULTILINE)
    _KEY_VALUE_RE     = re.compile(r"^\s*[\w\s]+\s*[:=]\s*\S", re.MULTILINE)
    _XML_TAG_RE       = re.compile(r"<([a-zA-Z][\w\-]*)>[^<]+</\1>")
    _TOOL_CALL_RE     = re.compile(
        r'(?:Action|Tool|Function|tool_name)\s*[:\-]\s*[\w\.]+', re.IGNORECASE
    )
    _CODE_FENCE_RE    = re.compile(r"```(\w*)\s*\n([\s\S]*?)```", re.IGNORECASE)

    # ─── Public API ──────────────────────────────────────────────────────

    def detect(self, text: str) -> OutputFormat:
        """Return the most likely OutputFormat for this text."""
        candidates = self.rank(text)
        return candidates[0].format if candidates else OutputFormat.UNKNOWN

    def rank(self, text: str) -> List[FormatCandidate]:
        """Return all plausible OutputFormat candidates sorted by confidence (desc)."""
        if not text or not text.strip():
            return [FormatCandidate(OutputFormat.UNKNOWN, 0.0, "empty input")]

        # First: check for code-fenced blocks — they give a strong signal
        fenced = self._detect_from_code_fence(text)
        if fenced and fenced.confidence >= 0.75:
            # Combine with raw analysis for corroboration
            raw = self._score_all(text)
            candidates = [fenced] + [c for c in raw if c.format != fenced.format]
        else:
            candidates = self._score_all(text)
            if fenced:
                candidates.insert(0, fenced)

        # Filter and sort
        filtered = [c for c in candidates if c.confidence >= self._MIN_CONFIDENCE]
        filtered.sort(key=lambda c: c.confidence, reverse=True)

        # Deduplicate by format
        seen: set = set()
        unique: List[FormatCandidate] = []
        for c in filtered:
            if c.format not in seen:
                seen.add(c.format)
                unique.append(c)

        return unique if unique else [FormatCandidate(OutputFormat.PLAIN_TEXT, 0.3, "fallback")]

    def detect_with_confidence(self, text: str) -> Tuple[OutputFormat, float]:
        """Return (format, confidence) for the best candidate."""
        candidates = self.rank(text)
        if not candidates:
            return OutputFormat.UNKNOWN, 0.0
        best = candidates[0]
        return best.format, best.confidence

    # ─── Private scoring ─────────────────────────────────────────────────

    def _score_all(self, text: str) -> List[FormatCandidate]:
        """Run all detectors and aggregate results."""
        detectors = [
            self._score_json,
            self._score_yaml,
            self._score_csv,
            self._score_markdown_table,
            self._score_numbered_list,
            self._score_bulleted_list,
            self._score_key_value,
            self._score_xml,
            self._score_tool_call,
            self._score_plain_text,
        ]
        results: List[FormatCandidate] = []
        for det in detectors:
            candidate = det(text)
            if candidate:
                results.append(candidate)
        return results

    def _detect_from_code_fence(self, text: str) -> Optional[FormatCandidate]:
        """Check for explicit language tags in markdown code fences."""
        match = self._CODE_FENCE_RE.search(text)
        if not match:
            return None
        lang = match.group(1).lower().strip()
        lang_map = {
            "json":       (OutputFormat.JSON, 0.95),
            "yaml":       (OutputFormat.YAML, 0.95),
            "yml":        (OutputFormat.YAML, 0.95),
            "csv":        (OutputFormat.CSV, 0.95),
            "xml":        (OutputFormat.XML, 0.90),
            "":           None,   # unlabelled — defer to content analysis
        }
        if lang in lang_map and lang_map[lang]:
            fmt, conf = lang_map[lang]
            return FormatCandidate(fmt, conf, f"code fence with lang={lang!r}")
        # Unlabelled fence — small boost only
        return FormatCandidate(OutputFormat.UNKNOWN, 0.0, "unlabelled code fence")

    def _score_json(self, text: str) -> Optional[FormatCandidate]:
        stripped = text.strip()
        # Try direct parse (the gold standard)
        try:
            json.loads(stripped)
            return FormatCandidate(OutputFormat.JSON, 0.99, "valid JSON parse succeeded")
        except json.JSONDecodeError:
            pass
        # Structural heuristics
        if (self._JSON_OBJECT_RE.search(stripped) or self._JSON_ARRAY_RE.search(stripped)):
            return FormatCandidate(OutputFormat.JSON, 0.65, "starts/ends with { } or [ ]")
        # Embedded JSON fragment
        if re.search(r'\{["\'][^"\']+["\']\s*:', stripped):
            return FormatCandidate(OutputFormat.JSON, 0.45, "contains JSON-like key pattern")
        return None

    def _score_yaml(self, text: str) -> Optional[FormatCandidate]:
        key_matches = len(self._YAML_KEY_RE.findall(text))
        list_matches = len(self._YAML_LIST_RE.findall(text))
        # Penalise if it looks like JSON (both use colons)
        json_penalty = 0.3 if (text.strip().startswith("{") or text.strip().startswith("[")) else 0.0
        score = 0.0
        evidence_parts = []
        if key_matches >= 3:
            score += 0.5
            evidence_parts.append(f"{key_matches} key: value lines")
        elif key_matches >= 1:
            score += 0.2
        if list_matches >= 2:
            score += 0.25
            evidence_parts.append(f"{list_matches} list items")
        score -= json_penalty
        if score > 0:
            return FormatCandidate(OutputFormat.YAML, score, ", ".join(evidence_parts) or "yaml-like structure")
        return None

    def _score_csv(self, text: str) -> Optional[FormatCandidate]:
        lines = [l for l in text.strip().splitlines() if l.strip()]
        if len(lines) < 2:
            return None
        header = lines[0]
        # Count consistent delimiter usage
        for delim in (",", ";", "\t", "|"):
            count_in_header = header.count(delim)
            if count_in_header < 1:
                continue
            # Check consistency across rows
            consistent = sum(
                1 for l in lines[1:min(5, len(lines))]
                if abs(l.count(delim) - count_in_header) <= 1
            )
            ratio = consistent / min(4, len(lines) - 1)
            if ratio >= 0.75:
                score = 0.5 + 0.3 * ratio
                return FormatCandidate(
                    OutputFormat.CSV, score,
                    f"delimiter={delim!r} consistent across {consistent}/{len(lines)-1} rows"
                )
        return None

    def _score_markdown_table(self, text: str) -> Optional[FormatCandidate]:
        if self._MD_TABLE_RE.search(text):
            return FormatCandidate(OutputFormat.MARKDOWN_TABLE, 0.92, "markdown table separator row found")
        if text.strip().startswith("|") and "|" in text:
            return FormatCandidate(OutputFormat.MARKDOWN_TABLE, 0.5, "pipe-delimited lines")
        return None

    def _score_numbered_list(self, text: str) -> Optional[FormatCandidate]:
        matches = self._NUMBERED_RE.findall(text)
        if len(matches) >= 3:
            return FormatCandidate(OutputFormat.NUMBERED_LIST, 0.85, f"{len(matches)} numbered items")
        if len(matches) >= 1:
            return FormatCandidate(OutputFormat.NUMBERED_LIST, 0.4, f"{len(matches)} numbered item(s)")
        return None

    def _score_bulleted_list(self, text: str) -> Optional[FormatCandidate]:
        matches = self._BULLETED_RE.findall(text)
        if len(matches) >= 3:
            return FormatCandidate(OutputFormat.BULLETED_LIST, 0.82, f"{len(matches)} bullet items")
        if len(matches) >= 1:
            return FormatCandidate(OutputFormat.BULLETED_LIST, 0.38, f"{len(matches)} bullet item(s)")
        return None

    def _score_key_value(self, text: str) -> Optional[FormatCandidate]:
        kv_matches = self._KEY_VALUE_RE.findall(text)
        # Discount if it's mostly YAML (avoid double-counting)
        yaml_list = self._YAML_LIST_RE.findall(text)
        if len(kv_matches) >= 2 and len(yaml_list) == 0:
            return FormatCandidate(OutputFormat.KEY_VALUE, 0.60, f"{len(kv_matches)} key: value pairs")
        return None

    def _score_xml(self, text: str) -> Optional[FormatCandidate]:
        matches = self._XML_TAG_RE.findall(text)
        if len(matches) >= 2:
            return FormatCandidate(OutputFormat.XML, 0.88, f"{len(matches)} XML tag pairs found")
        if re.search(r"<\w+\s*/?>", text):
            return FormatCandidate(OutputFormat.XML, 0.35, "XML-like tags detected")
        return None

    def _score_tool_call(self, text: str) -> Optional[FormatCandidate]:
        if self._TOOL_CALL_RE.search(text):
            # Check for argument block
            has_args = bool(re.search(r'(?:Input|Args|Arguments|Action Input)\s*[:\-]', text, re.I))
            score = 0.82 if has_args else 0.55
            return FormatCandidate(OutputFormat.TOOL_CALL, score, "tool/action pattern found")
        # OpenAI function call JSON pattern
        if re.search(r'"name"\s*:\s*"[^"]+"\s*,\s*"arguments"', text):
            return FormatCandidate(OutputFormat.TOOL_CALL, 0.90, "OpenAI function call JSON")
        return None

    def _score_plain_text(self, text: str) -> Optional[FormatCandidate]:
        """Always returns a baseline score for plain text."""
        # Score decreases as structural signals increase
        structure_signals = sum([
            bool(self._YAML_KEY_RE.search(text)),
            bool(self._NUMBERED_RE.search(text)),
            bool(self._BULLETED_RE.search(text)),
            text.strip().startswith(("{", "[")),
        ])
        score = max(0.1, 0.55 - 0.1 * structure_signals)
        return FormatCandidate(OutputFormat.PLAIN_TEXT, score, "default text fallback")
