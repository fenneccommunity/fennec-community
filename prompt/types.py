"""
Prompt Types & Shared Contracts
================================
All enumerations, dataclasses, and type aliases used across
the prompt orchestration engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional


# ─────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────

class PromptType(str, Enum):
    """Canonical prompt archetypes the engine can generate."""
    QA              = "qa"            # grounded question-answering
    CONVERSATIONAL  = "conversational"# multi-turn dialogue
    REASONING       = "reasoning"     # chain-of-thought / multi-hop
    AGENT           = "agent"         # ReAct / plan-and-execute
    TOOL_USE        = "tool_use"      # function-calling description
    SAFETY          = "safety"        # content moderation / guard
    SUMMARIZATION   = "summarization" # document summarization
    EXTRACTION      = "extraction"    # structured data extraction
    COMPARISON      = "comparison"    # compare / contrast multiple docs


class PromptStrategy(str, Enum):
    """Building strategy applied on top of the prompt type."""
    SIMPLE          = "simple"        # single-hop, straightforward
    MULTI_HOP       = "multi_hop"     # decompose → gather → synthesize
    CHAIN_OF_THOUGHT= "cot"           # show your reasoning step by step
    SELF_CONSISTENT = "self_consistent"# generate N answers, pick majority
    STEP_BACK       = "step_back"     # abstract first, then answer
    REACT           = "react"         # Reasoning + Acting (agent)
    LEAST_TO_MOST   = "least_to_most" # break into sub-questions


class OutputFormat(str, Enum):
    """Desired output format enforced by the engine."""
    TEXT            = "text"
    JSON            = "json"
    MARKDOWN        = "markdown"
    BULLET_LIST     = "bullet_list"
    STRUCTURED      = "structured"    # domain-specific schema
    CITATION        = "citation"      # answer + inline citations


class QueryComplexity(str, Enum):
    SIMPLE   = "simple"    # single fact
    MODERATE = "moderate"  # some reasoning needed
    COMPLEX  = "complex"   # multi-hop or multi-doc
    EXPERT   = "expert"    # deep domain knowledge


class UserProfile(str, Enum):
    GENERAL    = "general"
    TECHNICAL  = "technical"
    ACADEMIC   = "academic"
    EXECUTIVE  = "executive"   # brief, high-level


# ─────────────────────────────────────────────
# Core dataclasses
# ─────────────────────────────────────────────

@dataclass
class Document:
    """A retrieved passage / chunk used as context."""
    content:   str
    source:    str                = ""
    score:     float              = 1.0
    metadata:  Dict[str, Any]     = field(default_factory=dict)
    chunk_id:  Optional[str]      = None
    language:  str                = "en"


@dataclass
class Message:
    """A single turn in a conversation."""
    role:    Literal["system", "user", "assistant"]
    content: str


@dataclass
class PromptRequest:
    """
    Everything the engine needs to build an optimal prompt.
    Passed directly to ``PromptEngine.build()``.
    """
    # ── Required ──────────────────────────────────────────────────────
    query:      str

    # ── Context ───────────────────────────────────────────────────────
    documents:  List[Document]         = field(default_factory=list)
    memory:     List[Message]          = field(default_factory=list)

    # ── Intent / routing ─────────────────────────────────────────────
    prompt_type:    PromptType         = PromptType.QA
    strategy:       PromptStrategy     = PromptStrategy.SIMPLE
    output_format:  OutputFormat       = OutputFormat.TEXT
    complexity:     QueryComplexity    = QueryComplexity.SIMPLE
    user_profile:   UserProfile        = UserProfile.GENERAL

    # ── Output constraints ────────────────────────────────────────────
    max_context_tokens: int            = 3000
    max_answer_tokens:  int            = 512
    output_schema:  Optional[Dict]     = None    # JSON schema for structured output
    language:       str                = "en"

    # ── Feature flags ─────────────────────────────────────────────────
    enable_guardrails:   bool          = True
    enable_cot:          bool          = False   # auto-set by strategy
    enable_citations:    bool          = True
    enable_uncertainty:  bool          = True    # "say I don't know if unsure"

    # ── Metadata (passed through to result) ───────────────────────────
    session_id:  str                   = ""
    user_id:     str                   = ""
    trace_id:    str                   = ""
    extra:       Dict[str, Any]        = field(default_factory=dict)


@dataclass
class BuiltPrompt:
    """
    The result of ``PromptEngine.build()``.
    Contains the assembled prompt plus rich metadata for observability.
    """
    # ── The actual prompts ────────────────────────────────────────────
    system_prompt: str
    user_prompt:   str
    messages:      List[Message]       = field(default_factory=list)

    # ── Provenance ────────────────────────────────────────────────────
    prompt_type:   PromptType          = PromptType.QA
    strategy:      PromptStrategy      = PromptStrategy.SIMPLE
    output_format: OutputFormat        = OutputFormat.TEXT

    # ── Token accounting ─────────────────────────────────────────────
    estimated_tokens:    int           = 0
    context_tokens_used: int           = 0
    documents_included:  int           = 0
    documents_truncated: int           = 0

    # ── Guardrail flags ───────────────────────────────────────────────
    guardrails_applied:  List[str]     = field(default_factory=list)

    # ── Optimizer report ─────────────────────────────────────────────
    tokens_saved:        int           = 0
    optimization_notes:  List[str]     = field(default_factory=list)

    # ── Request echo ─────────────────────────────────────────────────
    session_id: str                    = ""
    trace_id:   str                    = ""

    def to_messages(self) -> List[Dict[str, str]]:
        """Return OpenAI-compatible messages list."""
        return [{"role": m.role, "content": m.content} for m in self.messages]

    def to_anthropic(self) -> Dict[str, Any]:
        """Return Anthropic-compatible payload fragment."""
        user_msgs = [m for m in self.messages if m.role != "system"]
        return {
            "system":   self.system_prompt,
            "messages": [{"role": m.role, "content": m.content} for m in user_msgs],
        }

    @property
    def full_text(self) -> str:
        """Combined system + user prompt as plain text (for debugging)."""
        return f"[SYSTEM]\n{self.system_prompt}\n\n[USER]\n{self.user_prompt}"
