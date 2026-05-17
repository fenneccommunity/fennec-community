"""
==================
ParallelChain — executes sub-chains concurrently with asyncio.gather
and aggregates results into a structured dict or list.


Example (named branches)
------------------------
>>> parallel = ParallelChain({
...     "summary":   summarize_chain,
...     "sentiment": sentiment_chain,
... })
>>> result = await parallel.arun("This movie was fantastic!")
>>> result["summary"]    # → "..."
>>> result["sentiment"]  # → "positive"

Example (list branches)
-----------------------
>>> parallel = ParallelChain([chain_a, chain_b, chain_c])
>>> result = await parallel.arun(data)   # → [res_a, res_b, res_c]
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union

from ..core.base import BaseChain
from ..core.config import ChainConfig
from ..core.context import ChainInput, ChainOutput
from ..core.tracing import ExecutionTracer

logger = logging.getLogger(__name__)


class ParallelChain(BaseChain):
    """
    Concurrent execution of multiple chains over the same input.

    Parameters
    ----------
    chains    : Either a dict {label: chain} or a list of chains.
    name      : Optional label.
    fail_fast : If True (default), raise as soon as any branch fails.
                If False, collect all results and include errors in output.
    config    : Retry / timeout / fallback settings.
    tracer    : Shared tracer.
    """

    def __init__(
        self,
        chains: Union[Dict[str, BaseChain], List[BaseChain]],
        name: str = "ParallelChain",
        fail_fast: bool = True,
        config: Optional[ChainConfig] = None,
        tracer: Optional[ExecutionTracer] = None,
    ) -> None:
        super().__init__(name=name, config=config, tracer=tracer)
        self.fail_fast = fail_fast

        if isinstance(chains, dict):
            self._named = True
            self._labels: List[str] = list(chains.keys())
            self._branches: List[BaseChain] = list(chains.values())
        else:
            self._named = False
            self._labels = [str(i) for i in range(len(chains))]
            self._branches = list(chains)

    # ------------------------------------------------------------------

    async def _aexecute(self, input_data: ChainInput) -> ChainOutput:
        """
        Launch all branches concurrently and aggregate results.
        """

        async def _run_branch(chain: BaseChain, label: str) -> Any:
            chain.tracer = self.tracer
            step_inp = ChainInput(
                data=input_data.data,
                metadata=dict(input_data.metadata),
                context=input_data.context,
            )
            out = await chain._aexecute(step_inp)
            if not out.success:
                raise RuntimeError(f"Branch '{label}' failed: {out.error}")
            return out.data

        if self.fail_fast:
            results = await asyncio.gather(
                *[_run_branch(c, lbl) for c, lbl in zip(self._branches, self._labels)]
            )
            aggregated: Any = (
                dict(zip(self._labels, results)) if self._named else list(results)
            )
        else:
            # Collect results even when some branches fail
            raw = await asyncio.gather(
                *[_run_branch(c, lbl) for c, lbl in zip(self._branches, self._labels)],
                return_exceptions=True,
            )
            aggregated = {}
            errors: Dict[str, str] = {}
            for lbl, res in zip(self._labels, raw):
                if isinstance(res, Exception):
                    errors[lbl] = str(res)
                    aggregated[lbl] = None
                else:
                    aggregated[lbl] = res

            if not self._named:
                aggregated = list(aggregated.values())

            if errors:
                logger.warning("[%s] some branches failed: %s", self.name, errors)
                return ChainOutput(
                    data=aggregated,
                    metadata={"branch_errors": errors},
                    success=len(errors) < len(self._branches),  # partial success
                )

        return ChainOutput(
            data=aggregated,
            metadata={"branch_count": len(self._branches)},
            success=True,
        )
