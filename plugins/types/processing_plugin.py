from __future__ import annotations
from abc import abstractmethod
from typing import Any, Dict, List
from ..base_plugin import BasePlugin
from ..metadata import ExecutionContext


class ProcessingPlugin(BasePlugin):
    """
    Plugins that transform, filter, rerank, or summarize data:
    rerankers, chunkers, extractors, translators, formatters.

    Input contract:  {"items": List[Any], ...extra}
    Output contract: List[Any]  (same or transformed items)
    """

    PLUGIN_TYPE = "processing"

    @abstractmethod
    async def execute(
        self,
        input_data: Dict[str, Any],
        context: ExecutionContext,
    ) -> List[Any]:
        """Process and return transformed items."""

    async def process(
        self,
        items: List[Any],
        context: ExecutionContext = None,
        **kwargs,
    ) -> List[Any]:
        """Convenience wrapper."""
        ctx = context or ExecutionContext()
        return await self.safe_execute({"items": items, **kwargs}, ctx)
