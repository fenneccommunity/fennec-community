"""chain_framework.core — foundational types."""

from .base import BaseChain
from .config import ChainConfig
from .context import ChainContext, ChainInput, ChainOutput
from .tracing import ExecutionTracer, TraceSpan

__all__ = [
    "BaseChain",
    "ChainConfig",
    "ChainContext",
    "ChainInput",
    "ChainOutput",
    "ExecutionTracer",
    "TraceSpan",
]
