"""
Prompt Building Strategies
===========================
Each strategy knows how to arrange system instructions, context,
memory, and user query into the optimal prompt shape for its purpose.

Available strategies
--------------------
SimpleStrategy          — direct single-hop QA
ChainOfThoughtStrategy  — explicit step-by-step reasoning
MultiHopStrategy        — decompose → sub-answer → synthesize
SelfConsistentStrategy  — generate N, pick best
StepBackStrategy        — abstract first, then ground answer
ReActStrategy           — Reasoning + Acting for agents
LeastToMostStrategy     — progressive sub-question decomposition
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import  Dict, List

from ..types import (
    BuiltPrompt,
    Message,
    PromptRequest,
    PromptStrategy,
    UserProfile,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────

class BaseStrategy(ABC):
    """Every strategy must implement build()."""

    STRATEGY: PromptStrategy  # set by subclass

    def build(
        self,
        request:        PromptRequest,
        context_block:  str,
        memory_block:   str,
        guardrail_block: str,
        citation_map:   Dict[int, str],
    ) -> BuiltPrompt:
        system = self._build_system(request, guardrail_block)
        user   = self._build_user(request, context_block, memory_block)
        messages = self._assemble_messages(request, system, user)

        from ..optimizer import PromptOptimizer
        opt = PromptOptimizer()
        system, user, saved, notes = opt.optimize(system, user, request)
        messages = self._assemble_messages(request, system, user)

        tokens = _estimate_tokens(system + " " + user)
        return BuiltPrompt(
            system_prompt        = system,
            user_prompt          = user,
            messages             = messages,
            prompt_type          = request.prompt_type,
            strategy             = self.STRATEGY,
            output_format        = request.output_format,
            estimated_tokens     = tokens,
            context_tokens_used  = _estimate_tokens(context_block),
            documents_included   = len(request.documents),
            tokens_saved         = saved,
            optimization_notes   = notes,
            session_id           = request.session_id,
            trace_id             = request.trace_id,
        )

    @abstractmethod
    def _build_system(self, request: PromptRequest, guardrail_block: str) -> str: ...

    @abstractmethod
    def _build_user(
        self, request: PromptRequest, context_block: str, memory_block: str
    ) -> str: ...

    def _assemble_messages(
        self, request: PromptRequest, system: str, user: str
    ) -> List[Message]:
        msgs: List[Message] = [Message(role="system", content=system)]
        # Inject prior memory as alternating user/assistant turns
        for m in request.memory[-10:]:
            msgs.append(m)
        msgs.append(Message(role="user", content=user))
        return msgs

    def _persona_line(self, request: PromptRequest) -> str:
        profile_map = {
            UserProfile.TECHNICAL:  "Use precise technical language; include relevant details.",
            UserProfile.ACADEMIC:   "Use formal academic language; cite evidence rigorously.",
            UserProfile.EXECUTIVE:  "Be extremely concise; highlight business impact first.",
            UserProfile.GENERAL:    "Use clear, accessible language; avoid unnecessary jargon.",
        }
        return profile_map.get(request.user_profile, "")

    def _language_line(self, request: PromptRequest) -> str:
        if request.language and request.language.lower() not in ("en", "english"):
            return f"Respond in {request.language}."
        return ""


def _estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)


# ─────────────────────────────────────────────
# 1. Simple Strategy
# ─────────────────────────────────────────────

class SimpleStrategy(BaseStrategy):
    """Direct single-hop question-answering."""

    STRATEGY = PromptStrategy.SIMPLE

    def _build_system(self, req: PromptRequest, guardrail_block: str) -> str:
        persona = self._persona_line(req)
        lang    = self._language_line(req)
        parts   = [
            "You are an expert AI assistant specialized in accurate, "
            "grounded question-answering.",
        ]
        if persona:
            parts.append(persona)
        if lang:
            parts.append(lang)
        if guardrail_block:
            parts.append(guardrail_block)
        return "\n\n".join(parts)

    def _build_user(self, req: PromptRequest, ctx: str, mem: str) -> str:
        parts: List[str] = []
        if ctx:
            parts.append(f"## Context\n{ctx}")
        if mem:
            parts.append(f"## Conversation History\n{mem}")
        parts.append(f"## Question\n{req.query}")
        parts.append("## Answer")
        return "\n\n".join(parts)


# ─────────────────────────────────────────────
# 2. Chain-of-Thought Strategy
# ─────────────────────────────────────────────

class ChainOfThoughtStrategy(BaseStrategy):
    """Explicit reasoning before the final answer."""

    STRATEGY = PromptStrategy.CHAIN_OF_THOUGHT

    def _build_system(self, req: PromptRequest, guardrail_block: str) -> str:
        parts = [
            "You are an expert AI assistant. You reason carefully and transparently "
            "before providing answers.",
            "Think step by step. Show your reasoning, then conclude with a clear final answer.",
        ]
        persona = self._persona_line(req)
        if persona:
            parts.append(persona)
        if self._language_line(req):
            parts.append(self._language_line(req))
        if guardrail_block:
            parts.append(guardrail_block)
        return "\n\n".join(parts)

    def _build_user(self, req: PromptRequest, ctx: str, mem: str) -> str:
        parts: List[str] = []
        if ctx:
            parts.append(f"## Context\n{ctx}")
        if mem:
            parts.append(f"## Conversation History\n{mem}")
        parts.append(f"## Question\n{req.query}")
        parts.append(
            "## Reasoning Process\n"
            "Think through the following:\n"
            "1. What is the question really asking?\n"
            "2. What relevant information exists in the context?\n"
            "3. Are there any conflicts or gaps in the context?\n"
            "4. What is the most accurate, well-supported answer?\n\n"
            "## Final Answer"
        )
        return "\n\n".join(parts)


# ─────────────────────────────────────────────
# 3. Multi-Hop Strategy
# ─────────────────────────────────────────────

class MultiHopStrategy(BaseStrategy):
    """
    Decompose complex questions into sub-questions, answer each,
    then synthesize a final grounded answer.
    """

    STRATEGY = PromptStrategy.MULTI_HOP

    def _build_system(self, req: PromptRequest, guardrail_block: str) -> str:
        parts = [
            "You are an expert AI assistant that excels at multi-hop reasoning "
            "over complex, information-rich contexts.",
            (
                "Approach:\n"
                "Step 1 — Decompose: Identify the sub-questions that must be "
                "answered to address the main question.\n"
                "Step 2 — Evidence: For each sub-question, find the relevant "
                "passage(s) in the context.\n"
                "Step 3 — Synthesize: Combine the sub-answers into a coherent, "
                "grounded final answer."
            ),
        ]
        if self._persona_line(req):
            parts.append(self._persona_line(req))
        if self._language_line(req):
            parts.append(self._language_line(req))
        if guardrail_block:
            parts.append(guardrail_block)
        return "\n\n".join(parts)

    def _build_user(self, req: PromptRequest, ctx: str, mem: str) -> str:
        parts: List[str] = []
        if ctx:
            parts.append(f"## Context Documents\n{ctx}")
        if mem:
            parts.append(f"## Conversation History\n{mem}")
        parts.append(f"## Main Question\n{req.query}")
        parts.append(
            "## Analysis\n"
            "**Sub-questions:** [List the key sub-questions]\n"
            "**Evidence for each:** [Quote or reference the relevant context]\n"
            "**Synthesis:** [Combine into a final answer]\n\n"
            "## Final Answer"
        )
        return "\n\n".join(parts)


# ─────────────────────────────────────────────
# 4. Self-Consistent Strategy
# ─────────────────────────────────────────────

class SelfConsistentStrategy(BaseStrategy):
    """
    Ask the model to generate multiple independent reasoning paths
    and converge on the most consistent answer.
    """

    STRATEGY = PromptStrategy.SELF_CONSISTENT

    def _build_system(self, req: PromptRequest, guardrail_block: str) -> str:
        parts = [
            "You are an expert AI assistant trained to reason with high accuracy "
            "by exploring multiple reasoning paths before committing to an answer.",
        ]
        if guardrail_block:
            parts.append(guardrail_block)
        return "\n\n".join(parts)

    def _build_user(self, req: PromptRequest, ctx: str, mem: str) -> str:
        parts: List[str] = []
        if ctx:
            parts.append(f"## Context\n{ctx}")
        parts.append(f"## Question\n{req.query}")
        parts.append(
            "## Task\n"
            "Reason through this question using THREE independent approaches:\n\n"
            "**Approach A:** [Reason from first principles]\n"
            "**Approach B:** [Reason from the most relevant evidence]\n"
            "**Approach C:** [Consider alternative interpretations]\n\n"
            "**Consensus:** Review all three approaches. Identify where they agree. "
            "If they disagree, identify the most well-supported answer.\n\n"
            "## Final Answer (most consistent conclusion)"
        )
        return "\n\n".join(parts)


# ─────────────────────────────────────────────
# 5. Step-Back Strategy
# ─────────────────────────────────────────────

class StepBackStrategy(BaseStrategy):
    """
    Generate a higher-level, abstract question first,
    then use that abstraction to ground the specific answer.
    """

    STRATEGY = PromptStrategy.STEP_BACK

    def _build_system(self, req: PromptRequest, guardrail_block: str) -> str:
        parts = [
            "You are an expert AI assistant that uses abstraction to improve "
            "reasoning accuracy. Before answering a specific question, you first "
            "identify the broader principle or concept involved.",
        ]
        if guardrail_block:
            parts.append(guardrail_block)
        return "\n\n".join(parts)

    def _build_user(self, req: PromptRequest, ctx: str, mem: str) -> str:
        parts: List[str] = []
        if ctx:
            parts.append(f"## Context\n{ctx}")
        parts.append(f"## Specific Question\n{req.query}")
        parts.append(
            "## Step 1 — Abstraction\n"
            "What is the more general principle, concept, or category "
            "that this question belongs to? State it clearly.\n\n"
            "## Step 2 — General Answer\n"
            "Answer the general/abstract version of the question using the context.\n\n"
            "## Step 3 — Specific Answer\n"
            "Now apply the general answer to the specific question asked.\n\n"
            "## Final Answer"
        )
        return "\n\n".join(parts)


# ─────────────────────────────────────────────
# 6. ReAct Strategy (Agent)
# ─────────────────────────────────────────────

class ReActStrategy(BaseStrategy):
    """
    Reasoning + Acting strategy for agentic tasks.
    Produces Thought / Action / Observation / ... / Final Answer loops.
    """

    STRATEGY = PromptStrategy.REACT

    def _build_system(self, req: PromptRequest, guardrail_block: str) -> str:
        parts = [
            "You are an autonomous AI agent. You reason and act iteratively to "
            "complete tasks. You have access to tools and can take actions.",
            (
                "Format:\n"
                "Thought: [your reasoning about what to do next]\n"
                "Action: [tool_name(arguments)]\n"
                "Observation: [result of the action]\n"
                "... (repeat as needed)\n"
                "Final Answer: [definitive answer to the original request]"
            ),
        ]
        if guardrail_block:
            parts.append(guardrail_block)
        return "\n\n".join(parts)

    def _build_user(self, req: PromptRequest, ctx: str, mem: str) -> str:
        parts: List[str] = []
        if ctx:
            parts.append(f"## Available Information\n{ctx}")
        if req.extra.get("tools"):
            tools_text = _format_tools(req.extra["tools"])
            parts.append(f"## Available Tools\n{tools_text}")
        if mem:
            parts.append(f"## Prior Conversation\n{mem}")
        parts.append(f"## Task\n{req.query}")
        parts.append("Thought:")
        return "\n\n".join(parts)


# ─────────────────────────────────────────────
# 7. Least-to-Most Strategy
# ─────────────────────────────────────────────

class LeastToMostStrategy(BaseStrategy):
    """
    Solve simpler sub-problems first, then build up to the full answer.
    Excellent for multi-step mathematical or logical questions.
    """

    STRATEGY = PromptStrategy.LEAST_TO_MOST

    def _build_system(self, req: PromptRequest, guardrail_block: str) -> str:
        parts = [
            "You are an expert AI assistant that solves complex problems by "
            "breaking them into progressively simpler sub-problems and solving "
            "them in order from simplest to most complex.",
        ]
        if guardrail_block:
            parts.append(guardrail_block)
        return "\n\n".join(parts)

    def _build_user(self, req: PromptRequest, ctx: str, mem: str) -> str:
        parts: List[str] = []
        if ctx:
            parts.append(f"## Context\n{ctx}")
        parts.append(f"## Problem\n{req.query}")
        parts.append(
            "## Solution\n"
            "**Step 1 — Decompose:** What are the simplest sub-problems?\n"
            "[List them from simplest to most complex]\n\n"
            "**Step 2 — Solve each:**\n"
            "[Solve sub-problem 1] → [Result 1]\n"
            "[Solve sub-problem 2 using Result 1] → [Result 2]\n"
            "... and so on.\n\n"
            "**Step 3 — Final Answer:**"
        )
        return "\n\n".join(parts)


# ─────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────

STRATEGY_REGISTRY: Dict[PromptStrategy, BaseStrategy] = {
    PromptStrategy.SIMPLE:          SimpleStrategy(),
    PromptStrategy.CHAIN_OF_THOUGHT: ChainOfThoughtStrategy(),
    PromptStrategy.MULTI_HOP:       MultiHopStrategy(),
    PromptStrategy.SELF_CONSISTENT: SelfConsistentStrategy(),
    PromptStrategy.STEP_BACK:       StepBackStrategy(),
    PromptStrategy.REACT:           ReActStrategy(),
    PromptStrategy.LEAST_TO_MOST:  LeastToMostStrategy(),
}


def get_strategy(strategy: PromptStrategy) -> BaseStrategy:
    impl = STRATEGY_REGISTRY.get(strategy)
    if impl is None:
        logger.warning("[Strategy] Unknown strategy '%s', falling back to Simple.", strategy)
        return STRATEGY_REGISTRY[PromptStrategy.SIMPLE]
    return impl


# ─────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────

def _format_tools(tools: List[Dict]) -> str:
    lines = []
    for t in tools:
        name = t.get("name", "?")
        desc = t.get("description", "")
        params = t.get("parameters", {})
        lines.append(f"- **{name}**: {desc}")
        if params:
            import json
            lines.append(f"  Parameters: {json.dumps(params, separators=(',', ':'))}")
    return "\n".join(lines)
