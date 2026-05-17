"""
RAG Plugin Framework
====================
AI-ready, modular, production-grade plugin ecosystem for RAG systems.

Quick-start
-----------
    from fennec_community.plugins import PluginManager, PluginMetadata, SchemaProperty
    from fennec_community.plugins import RetrievalPlugin, ToolPlugin

    manager = PluginManager(plugin_dirs=["./my_plugins"])
    await manager.startup()

    # Execute a plugin directly
    result = await manager.execute("web_search", {"query": "RAG 2025"}, context)

    # Let the AI selector pick the best plugin
    plugin = manager.select_one("find recent papers on LLMs")
    result = await manager.execute(plugin.name, {"query": "..."}, context)

    # Get LLM tool descriptors
    tools = manager.tool_descriptors()   # OpenAI format
    tools = manager.anthropic_tools()    # Anthropic format
"""

from .base_plugin import (
    BasePlugin,
    PluginConfig,
    PluginError,
    PluginPermissionError,
    PluginTimeoutError,
    PluginValidationError,
)
from .metadata import (
    CostTier,
    ExecutionContext,
    PermissionType,
    PluginMetadata,
    PluginPriority,
    PluginStatus,
    SchemaProperty,
)
from .plugin_manager import AIPluginSelector, PluginManager, PluginObservability
from .registry import PluginRegistry
from .loader import PluginLoader
from .security import ExecutionSandbox, InputSanitizer, PermissionGuard
from .types import (
    ActionPlugin,
    ProcessingPlugin,
    RetrievalPlugin,
    ToolPlugin,
    PLUGIN_TYPE_MAP,
)

__all__ = [
    # Core
    "PluginManager",
    "PluginRegistry",
    "PluginLoader",
    # Base
    "BasePlugin",
    "PluginConfig",
    # Errors
    "PluginError",
    "PluginTimeoutError",
    "PluginValidationError",
    "PluginPermissionError",
    # Metadata
    "PluginMetadata",
    "SchemaProperty",
    "ExecutionContext",
    "PluginStatus",
    "PluginPriority",
    "PermissionType",
    "CostTier",
    # Plugin types
    "RetrievalPlugin",
    "ToolPlugin",
    "ActionPlugin",
    "ProcessingPlugin",
    "PLUGIN_TYPE_MAP",
    # Security
    "PermissionGuard",
    "InputSanitizer",
    "ExecutionSandbox",
    # AI
    "AIPluginSelector",
    "PluginObservability",
]

__version__ = "2.0.0"
