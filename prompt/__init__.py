"""
RAG Prompt Orchestration Engine
================================
AI-ready, dynamic, context-aware prompt engineering for RAG systems.

Quick-start
-----------
    from fennec_community.prompt import PromptEngine, Document

    engine = PromptEngine()

    prompt = engine.build(
        query     = "What caused the 2008 financial crisis?",
        documents = [
            {"content": "...", "source": "wiki", "score": 0.92},
        ],
        strategy  = "multi_hop",
        output_format = "json",
    )

    # OpenAI
    response = openai_client.chat.completions.create(
        model    = "gpt-4o",
        messages = prompt.to_messages(),
    )

    # Anthropic
    payload  = prompt.to_anthropic()
    response = anthropic_client.messages.create(**payload, model="claude-opus-4-20250514")
"""

from .prompt_engine import PromptEngine, PromptMetrics, FeedbackEntry
from .builder import PromptBuilder
from .context_manager import ContextManager, ContextResult
from .guardrails import Guardrail, GuardrailEngine, GuardrailLibrary
from .optimizer import PromptOptimizer
from .types import (
    BuiltPrompt,
    Document,
    Message,
    OutputFormat,
    PromptRequest,
    PromptStrategy,
    PromptType,
    QueryComplexity,
    UserProfile,
)
from .strategies import get_strategy, STRATEGY_REGISTRY

__all__ = [
    # Engine
    "PromptEngine",
    "PromptMetrics",
    "FeedbackEntry",
    # Builder
    "PromptBuilder",
    # Context
    "ContextManager",
    "ContextResult",
    # Guardrails
    "Guardrail",
    "GuardrailEngine",
    "GuardrailLibrary",
    # Optimizer
    "PromptOptimizer",
    # Types
    "PromptRequest",
    "BuiltPrompt",
    "Document",
    "Message",
    "PromptType",
    "PromptStrategy",
    "OutputFormat",
    "QueryComplexity",
    "UserProfile",
    # Strategies
    "get_strategy",
    "STRATEGY_REGISTRY",
]

__version__ = "2.0.0"
