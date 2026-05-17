"""
====================
SequentialChain — executes sub-chains one after another, piping the
output of each step into the input of the next.


Example
-------
>>> clean   = TransformChain(str.strip)
>>> upper   = TransformChain(str.upper)
>>> pipeline = clean >> upper          # or SequentialChain([clean, upper])
>>> await pipeline.arun("  hello  ")
'HELLO'
"""

from __future__ import annotations

import logging
from typing import List, Optional

from ..core.base import BaseChain
from ..core.config import ChainConfig
from ..core.context import ChainInput, ChainOutput
from ..core.tracing import ExecutionTracer

logger = logging.getLogger(__name__)


class SequentialChain(BaseChain):
    """
    Runs a list of chains sequentially, wiring outputs to inputs.

    Parameters
    ----------
    chains : Ordered list of chains to execute.
    name   : Optional label.
    config : Retry / timeout / fallback settings.
    tracer : Shared tracer (inherited by sub-chains when set).

    Usage
    -----
    >>> pipeline = SequentialChain([step1, step2, step3])
    >>> result   = await pipeline.arun(input_data)

    Operator shorthand
    ------------------
    >>> pipeline = step1 >> step2 >> step3
    """

    def __init__(
        self,
        chains: List[BaseChain],
        name: str = "SequentialChain",
        config: Optional[ChainConfig] = None,
        tracer: Optional[ExecutionTracer] = None,
    ) -> None:
        super().__init__(name=name, config=config, tracer=tracer)
        self.chains: List[BaseChain] = list(chains)

    # ------------------------------------------------------------------

    async def _aexecute(self, input_data: ChainInput) -> ChainOutput:
        """
        Pipe data through each chain in order.
        """
        current_data = input_data.data
        accumulated_meta = dict(input_data.metadata)
        ctx = input_data.context

        for idx, chain in enumerate(self.chains):
            logger.debug("[%s] step %d/%d → %s", self.name, idx + 1, len(self.chains), chain.name)

            step_input = ChainInput(
                data=current_data,
                metadata=accumulated_meta,
                context=ctx,
            )

            # Propagate the shared tracer so child spans appear in the tree
            chain.tracer = self.tracer

            step_output = await chain._aexecute(step_input)

            if not step_output.success:
                return ChainOutput(
                    data=None,
                    success=False,
                    error=f"[{chain.name}] step {idx + 1} failed: {step_output.error}",
                    metadata=accumulated_meta,
                )

            current_data = step_output.data
            accumulated_meta.update(step_output.metadata)

        return ChainOutput(data=current_data, metadata=accumulated_meta, success=True)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def add(self, chain: BaseChain) -> "SequentialChain":
        """Append a chain and return self for chaining calls."""
        self.chains.append(chain)
        return self

    def __len__(self) -> int:
        return len(self.chains)
