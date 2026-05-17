"""
=====================
ConditionalChain — branches execution based on a condition function.
TransformChain    — applies a pure function to the input data.


ConditionalChain example
------------------------
>>> def route(data):
...     return "positive" if data["score"] > 0.5 else "negative"
...
>>> chain = ConditionalChain(
...     condition=route,
...     branches={
...         "positive": positive_chain,
...         "negative": negative_chain,
...     },
...     default=fallback_chain,
... )
>>> result = await chain.arun({"score": 0.8})

TransformChain example
----------------------
>>> clean = TransformChain(str.strip, name="strip_whitespace")
>>> upper = TransformChain(str.upper, name="to_upper")
>>> pipeline = clean >> upper
>>> await pipeline.arun("  hello  ")
'HELLO'
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional
from ..core.base import BaseChain
from ..core.config import ChainConfig
from ..core.context import ChainInput, ChainOutput
from ..core.tracing import ExecutionTracer

logger = logging.getLogger(__name__)


# ============================================================================
# ConditionalChain
# ============================================================================


class ConditionalChain(BaseChain):
    """
    Routes execution to one of several branches based on a condition.

    Parameters
    ----------
    condition : Callable that receives ``input_data.data`` and returns
                a string key matching one of the ``branches``.
    branches  : Dict mapping string keys to chains.
    default   : Fallback chain when the condition key is not in branches.
                If None and the key is missing, an error is returned.
    name      : Optional label.
    config    : Per-chain settings.
    tracer    : Shared tracer.
    """

    def __init__(
        self,
        condition: Callable[[Any], str],
        branches: Dict[str, BaseChain],
        default: Optional[BaseChain] = None,
        name: str = "ConditionalChain",
        config: Optional[ChainConfig] = None,
        tracer: Optional[ExecutionTracer] = None,
    ) -> None:
        super().__init__(name=name, config=config, tracer=tracer)
        self.condition = condition
        self.branches = branches
        self.default = default

    async def _aexecute(self, input_data: ChainInput) -> ChainOutput:
        """
        Evaluate condition → select branch → execute.
        """
        try:
            route_key = self.condition(input_data.data)
        except Exception as exc:
            return ChainOutput(
                data=None,
                success=False,
                error=f"Condition evaluation failed: {exc}",
            )

        selected = self.branches.get(route_key, self.default)
        if selected is None:
            return ChainOutput(
                data=None,
                success=False,
                error=f"No branch for route key '{route_key}' and no default set.",
            )

        logger.debug("[%s] routing → '%s'", self.name, route_key)
        selected.tracer = self.tracer
        out = await selected._aexecute(input_data)
        out.metadata["selected_route"] = route_key
        return out


# ============================================================================
# TransformChain
# ============================================================================


class TransformChain(BaseChain):
    """
    Applies a synchronous or async transformation function to the input.

    Parameters
    ----------
    transform_fn : A ``Callable[[Any], Any]`` (sync or async).
    name         : Optional label.
    config       : Per-chain settings.
    tracer       : Shared tracer.

    Example
    -------
    >>> import json
    >>> parse = TransformChain(json.loads, name="json_parser")
    >>> result = await parse.arun('{"key": 1}')
    >>> result
    {'key': 1}
    """

    def __init__(
        self,
        transform_fn: Callable[[Any], Any],
        name: Optional[str] = None,
        config: Optional[ChainConfig] = None,
        tracer: Optional[ExecutionTracer] = None,
    ) -> None:
        super().__init__(
            name=name or getattr(transform_fn, "__name__", "TransformChain"),
            config=config,
            tracer=tracer,
        )
        self.transform_fn = transform_fn

    async def _aexecute(self, input_data: ChainInput) -> ChainOutput:
        """
        Apply transform_fn to input_data.data.
        """
        import asyncio

        try:
            if asyncio.iscoroutine(self.transform_fn):
                result = await self.transform_fn(input_data.data)
            else:
                result = self.transform_fn(input_data.data)
            return ChainOutput(data=result, success=True)
        except Exception as exc:
            return ChainOutput(
                data=None,
                success=False,
                error=f"TransformChain '{self.name}' raised: {exc}",
            )
