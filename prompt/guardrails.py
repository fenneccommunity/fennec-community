"""
Prompt Guardrails
=================
Injects explicit safety, grounding, and output-quality instructions
directly into the prompt — the cheapest and most reliable way to
reduce hallucination, scope drift, and unsafe outputs.

Design principle
----------------
Guardrails are NOT post-processing filters; they are prompt instructions
that steer the model BEFORE it generates. Each guardrail is:
  • Named (for observability)
  • Conditional (only applied when relevant)
  • Composable (multiple guardrails stack cleanly)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .types import OutputFormat, PromptRequest, PromptType, QueryComplexity

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Guardrail base
# ─────────────────────────────────────────────

@dataclass
class Guardrail:
    name:        str
    instruction: str
    priority:    int  = 50   # higher = injected first


# ─────────────────────────────────────────────
# Guardrail Library
# ─────────────────────────────────────────────

class GuardrailLibrary:
    """All available guardrail instructions, indexed by name."""

    # ── Anti-hallucination ─────────────────────────────────────────────
    GROUNDING = Guardrail(
        name        = "grounding",
        priority    = 100,
        instruction = (
            "Answer ONLY using the information contained in the provided context. "
            "Do NOT use any prior knowledge, make assumptions, or extrapolate beyond "
            "what is explicitly stated in the context."
        ),
    )

    UNCERTAINTY = Guardrail(
        name        = "uncertainty",
        priority    = 90,
        instruction = (
            "If the context does not contain sufficient information to answer the question "
            "confidently, respond with: \"I don't have enough information in the provided "
            "context to answer this question accurately.\" Do NOT guess or fabricate."
        ),
    )

    NO_FABRICATION = Guardrail(
        name        = "no_fabrication",
        priority    = 95,
        instruction = (
            "Never invent facts, statistics, names, dates, or citations. "
            "Every claim in your answer must be traceable to a specific passage "
            "in the provided context."
        ),
    )

    # ── Citation ──────────────────────────────────────────────────────
    CITE_SOURCES = Guardrail(
        name        = "cite_sources",
        priority    = 80,
        instruction = (
            "When making a factual claim, cite the source using its bracketed number "
            "(e.g., 'According to [1]...' or '...as stated in [2].'). "
            "Do not cite a source for information that does not appear in that passage."
        ),
    )

    # ── Scope ─────────────────────────────────────────────────────────
    STAY_ON_TOPIC = Guardrail(
        name        = "stay_on_topic",
        priority    = 70,
        instruction = (
            "Stay strictly within the scope of the user's question. "
            "Do not volunteer unrelated information, tangential opinions, "
            "or advice that was not requested."
        ),
    )

    NO_PERSONAL_OPINIONS = Guardrail(
        name        = "no_personal_opinions",
        priority    = 60,
        instruction = (
            "Do not express personal opinions, beliefs, or preferences. "
            "Report what the context says; do not editorialize."
        ),
    )

    # ── Safety ────────────────────────────────────────────────────────
    SAFE_OUTPUT = Guardrail(
        name        = "safe_output",
        priority    = 110,
        instruction = (
            "Do not produce content that is harmful, offensive, discriminatory, "
            "or violates ethical guidelines. If the question implies such content, "
            "decline politely and explain why."
        ),
    )

    PII_PROTECTION = Guardrail(
        name        = "pii_protection",
        priority    = 105,
        instruction = (
            "Do not expose, repeat, or infer any personally identifiable information "
            "(PII) such as names, addresses, phone numbers, email addresses, "
            "or financial details from the context."
        ),
    )

    # ── Format ────────────────────────────────────────────────────────
    CONCISE = Guardrail(
        name        = "concise",
        priority    = 40,
        instruction = (
            "Be concise. Avoid unnecessary preamble, repetition, or filler phrases "
            "like 'Great question!' or 'Certainly!'. Get straight to the answer."
        ),
    )

    NO_MARKDOWN_LEAKAGE = Guardrail(
        name        = "no_markdown_leakage",
        priority    = 30,
        instruction = (
            "Do not include markdown formatting (bold, italic, code blocks) "
            "unless the output format explicitly requires it."
        ),
    )

    # ── Reasoning quality ─────────────────────────────────────────────
    SHOW_REASONING = Guardrail(
        name        = "show_reasoning",
        priority    = 50,
        instruction = (
            "Think step by step before giving your final answer. "
            "Show your reasoning process clearly, then conclude with a direct answer."
        ),
    )

    SELF_CHECK = Guardrail(
        name        = "self_check",
        priority    = 45,
        instruction = (
            "Before finalising your answer, verify: "
            "(1) Is every claim supported by the context? "
            "(2) Have I answered the actual question asked? "
            "(3) Is the answer complete and accurate?"
        ),
    )


# ─────────────────────────────────────────────
# Guardrail Engine
# ─────────────────────────────────────────────

class GuardrailEngine:
    """
    Selects and assembles guardrail instructions for a given PromptRequest,
    then formats them into a single injected block.

    Usage
    -----
    engine = GuardrailEngine()
    block, applied = engine.build(request)
    # inject ``block`` into the system prompt
    """

    # Map from OutputFormat → format-specific instruction
    _FORMAT_INSTRUCTIONS: Dict[OutputFormat, str] = {
        OutputFormat.JSON: (
            'Return your answer as a valid JSON object. '
            'Schema: {{"answer": "<string>", "sources": ["<source_id>"], '
            '"confidence": <0.0-1.0>, "reasoning": "<optional>"}}\n'
            'Output ONLY the JSON object. No preamble, no markdown fences.'
        ),
        OutputFormat.BULLET_LIST: (
            "Format your answer as a concise bulleted list. "
            "Each bullet should be a self-contained, factual statement."
        ),
        OutputFormat.MARKDOWN: (
            "Format your answer using Markdown. Use headers, bullet points, "
            "and bold text where appropriate to improve readability."
        ),
        OutputFormat.CITATION: (
            "Format your answer as flowing prose with inline citations in "
            "square-bracket format, e.g. [1], [2]. "
            "End with a 'Sources' section listing each cited document."
        ),
        OutputFormat.STRUCTURED: (
            "Return your answer as a structured JSON object matching the provided schema. "
            "Every field in the schema is required unless marked optional."
        ),
    }

    def __init__(self, extra_guardrails: Optional[List[Guardrail]] = None) -> None:
        self._extra = extra_guardrails or []

    def build(self, request: PromptRequest) -> tuple[str, List[str]]:
        """
        Select applicable guardrails and render the instruction block.

        Returns
        -------
        (instruction_block, list_of_applied_guardrail_names)
        """
        guardrails = self._select(request)
        guardrails += self._extra
        # Sort by priority (highest first)
        guardrails = sorted(guardrails, key=lambda g: g.priority, reverse=True)
        # Deduplicate by name
        seen: set = set()
        unique: List[Guardrail] = []
        for g in guardrails:
            if g.name not in seen:
                seen.add(g.name)
                unique.append(g)

        block = self._render(unique, request)
        applied = [g.name for g in unique]
        logger.debug("[Guardrails] Applied: %s", applied)
        return block, applied

    # ─────────────────────────────────────────
    # Selection logic
    # ─────────────────────────────────────────

    def _select(self, req: PromptRequest) -> List[Guardrail]:
        lib = GuardrailLibrary
        selected: List[Guardrail] = []

        # Always-on
        selected.append(lib.SAFE_OUTPUT)
        selected.append(lib.CONCISE)

        if req.enable_guardrails:
            # Grounding — whenever we have context documents
            if req.documents:
                selected.append(lib.GROUNDING)
                selected.append(lib.NO_FABRICATION)

            # Uncertainty acknowledgement
            if req.enable_uncertainty:
                selected.append(lib.UNCERTAINTY)

            # Citations
            if req.enable_citations and req.documents:
                selected.append(lib.CITE_SOURCES)

            # Scope
            if req.prompt_type not in (PromptType.AGENT, PromptType.TOOL_USE):
                selected.append(lib.STAY_ON_TOPIC)

            # PII protection for user-facing RAG
            selected.append(lib.PII_PROTECTION)

        # Strategy-driven
        from .types import PromptStrategy
        if req.strategy in (
            PromptStrategy.CHAIN_OF_THOUGHT,
            PromptStrategy.MULTI_HOP,
            PromptStrategy.LEAST_TO_MOST,
        ):
            selected.append(lib.SHOW_REASONING)

        if req.complexity in (QueryComplexity.COMPLEX, QueryComplexity.EXPERT):
            selected.append(lib.SELF_CHECK)

        return selected

    # ─────────────────────────────────────────
    # Rendering
    # ─────────────────────────────────────────

    def _render(self, guardrails: List[Guardrail], req: PromptRequest) -> str:
        lines: List[str] = []

        if guardrails:
            lines.append("## Instructions & Constraints")
            for i, g in enumerate(guardrails, 1):
                lines.append(f"{i}. {g.instruction}")

        # Output format instruction
        fmt_instr = self._FORMAT_INSTRUCTIONS.get(req.output_format)
        if fmt_instr:
            lines.append("\n## Output Format")
            # Inject custom schema if provided
            if req.output_schema and req.output_format == OutputFormat.STRUCTURED:
                import json
                schema_str = json.dumps(req.output_schema, indent=2)
                fmt_instr += f"\n\nSchema:\n```json\n{schema_str}\n```"
            lines.append(fmt_instr)

        return "\n".join(lines)
