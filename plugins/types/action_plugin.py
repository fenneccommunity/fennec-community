from __future__ import annotations
from abc import abstractmethod
from typing import Any, Dict
from ..base_plugin import BasePlugin
from ..metadata import ExecutionContext




class ActionPlugin(BasePlugin):
    """
    Plugins that perform side-effects / mutations:
    send email, write to DB, update calendar, call webhook, etc.

    Action plugins require explicit confirmation (can_execute check)
    unless the context carries a bypass flag.
    """

    PLUGIN_TYPE = "action"
    REQUIRES_CONFIRMATION: bool = True   # set False if safe to auto-execute

    async def can_execute(
        self, input_data: Dict[str, Any], context: ExecutionContext
    ) -> bool:
        """
        Override to add pre-flight checks (auth, quota, dry-run mode).
        Default: True (always allowed).
        """
        return True

    async def safe_execute(
        self,
        input_data: Dict[str, Any],
        context: ExecutionContext,
    ) -> Any:
        if not await self.can_execute(input_data, context):
            from ..base_plugin import PluginPermissionError
            raise PluginPermissionError(
                f"[{self.name}] Pre-flight check failed; execution blocked."
            )
        return await super().safe_execute(input_data, context)

    @abstractmethod
    async def execute(
        self,
        input_data: Dict[str, Any],
        context: ExecutionContext,
    ) -> Dict[str, Any]:
        """
        Execute the action.
        Return dict with at least "success": bool and "message": str.
        """