from __future__ import annotations
from abc import abstractmethod
from typing import Any, Dict
from ..base_plugin import BasePlugin
from ..metadata import ExecutionContext


class ToolPlugin(BasePlugin):
    """
    Plugins that call external tools, APIs, or utilities:
    web search, calculators, code interpreters, SQL runners, etc.

    These are the primary target for LLM function-calling.
    """

    PLUGIN_TYPE = "tool"

    @abstractmethod
    async def execute(
        self,
        input_data: Dict[str, Any],
        context: ExecutionContext,
    ) -> Dict[str, Any]:
        """Run the tool and return a structured result."""