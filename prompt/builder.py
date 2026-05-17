"""
Prompt Builder
==============
Sits between the PromptEngine and the strategy layer.
Resolves the correct strategy for each (prompt_type, strategy) pair,
injects type-specific system persona overrides, and delegates
to the strategy's build() method.

Each prompt type has a canonical default strategy; callers can override.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from .context_manager import ContextManager, ContextResult
from .guardrails import GuardrailEngine
from .strategies import STRATEGY_REGISTRY, get_strategy
from .types import (
    BuiltPrompt,
    PromptRequest,
    PromptStrategy,
    PromptType,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Default strategy per prompt type
# ─────────────────────────────────────────────

_DEFAULT_STRATEGY: Dict[PromptType, PromptStrategy] = {
    PromptType.QA:             PromptStrategy.SIMPLE,
    PromptType.CONVERSATIONAL: PromptStrategy.SIMPLE,
    PromptType.REASONING:      PromptStrategy.CHAIN_OF_THOUGHT,
    PromptType.AGENT:          PromptStrategy.REACT,
    PromptType.TOOL_USE:       PromptStrategy.REACT,
    PromptType.SAFETY:         PromptStrategy.SIMPLE,
    PromptType.SUMMARIZATION:  PromptStrategy.CHAIN_OF_THOUGHT,
    PromptType.EXTRACTION:     PromptStrategy.SIMPLE,
    PromptType.COMPARISON:     PromptStrategy.MULTI_HOP,
}

# Auto-upgrade to CoT for complex / expert queries
_COMPLEXITY_UPGRADE: Dict[str, PromptStrategy] = {
    "complex": PromptStrategy.CHAIN_OF_THOUGHT,
    "expert":  PromptStrategy.MULTI_HOP,
}


class PromptBuilder:
    """
    Orchestrates context engineering, guardrail selection, and strategy
    dispatch to produce a fully-assembled BuiltPrompt.
    """

    def __init__(
        self,
        context_manager: Optional[ContextManager] = None,
        guardrail_engine: Optional[GuardrailEngine] = None,
    ) -> None:
        self.context_manager  = context_manager  or ContextManager()
        self.guardrail_engine = guardrail_engine or GuardrailEngine()

    def build(self, request: PromptRequest) -> BuiltPrompt:
        """
        Full build pipeline:
          1. Resolve effective strategy (type default + complexity upgrade)
          2. Build context block
          3. Build memory block
          4. Build guardrail block
          5. Dispatch to strategy
        """
        # 1. Strategy resolution
        effective_strategy = self._resolve_strategy(request)
        request = _replace(request, strategy=effective_strategy)

        # 2. Context engineering
        ctx_result: ContextResult = self.context_manager.build(request)

        # 3. Memory formatting
        memory_block = self.context_manager.format_memory(request.memory)

        # 4. Guardrails
        guardrail_block, applied = self.guardrail_engine.build(request)

        # 5. Strategy dispatch
        strategy_impl = get_strategy(effective_strategy)
        prompt = strategy_impl.build(
            request        = request,
            context_block  = ctx_result.context_block,
            memory_block   = memory_block,
            guardrail_block = guardrail_block,
            citation_map   = ctx_result.citation_map,
        )

        # Enrich result with guardrail + context metadata
        prompt.guardrails_applied  = applied
        prompt.documents_included  = len(ctx_result.included_docs)
        prompt.documents_truncated = len(ctx_result.excluded_docs)
        prompt.context_tokens_used = ctx_result.tokens_used

        logger.info(
            "[Builder] Built %s/%s prompt | %d docs | %d tokens | %d guardrails",
            request.prompt_type.value,
            effective_strategy.value,
            prompt.documents_included,
            prompt.estimated_tokens,
            len(applied),
        )
        return prompt

    # ─────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────

    @staticmethod
    def _resolve_strategy(request: PromptRequest) -> PromptStrategy:
        # Explicit caller choice
        if request.strategy != PromptStrategy.SIMPLE:
            return request.strategy

        # Type default
        base = _DEFAULT_STRATEGY.get(request.prompt_type, PromptStrategy.SIMPLE)

        # Complexity upgrade (only if not already a strong strategy)
        if base == PromptStrategy.SIMPLE:
            upgrade = _COMPLEXITY_UPGRADE.get(request.complexity.value)
            if upgrade:
                return upgrade

        return base


# ─────────────────────────────────────────────
# Immutable-ish request modifier (dataclass replace)
# ─────────────────────────────────────────────

def _replace(request: PromptRequest, **kwargs) -> PromptRequest:
    """Return a shallow copy of request with specified fields overridden."""
    from dataclasses import replace
    return replace(request, **kwargs)
