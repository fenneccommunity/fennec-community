from __future__ import annotations
from ..base_plugin import BasePlugin
from ..metadata import ExecutionContext
from abc import abstractmethod
from typing import Any, Dict, List

# ─────────────────────────────────────────────
# 1. Retrieval Plugin
# ─────────────────────────────────────────────

class RetrievalPlugin(BasePlugin):
    """
    Plugins that retrieve information from a source:
    vector stores, search engines, databases, knowledge bases.

    Output contract
    ---------------
    execute() must return a list of dicts with at least:
      - "content": str
      - "score":   float   (relevance 0-1)
      - "source":  str
    """

    PLUGIN_TYPE = "retrieval"

    @abstractmethod
    async def execute(
        self,
        input_data: Dict[str, Any],
        context: ExecutionContext,
    ) -> List[Dict[str, Any]]:
        """Return ranked list of retrieved documents."""

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        context: ExecutionContext = None,
    ) -> List[Dict[str, Any]]:
        """Convenience wrapper; calls safe_execute internally."""
        ctx = context or ExecutionContext(query=query)
        return await self.safe_execute({"query": query, "top_k": top_k}, ctx)
