"""
Plugin Type Hierarchy
=====================
Four canonical plugin types for a RAG ecosystem:

  RetrievalPlugin   — fetch / search documents or passages
  ToolPlugin        — call external APIs / utilities
  ActionPlugin      — write / mutate state in the world
  ProcessingPlugin  — transform, rank, rerank, summarize data

All types extend BasePlugin. Agents can filter by type tag.
"""
from .action_plugin import ActionPlugin
from .processing_plugin import ProcessingPlugin
from .retrieval_plugin import RetrievalPlugin
from .tool_plugin import ToolPlugin

# ─────────────────────────────────────────────
# Registry of all type classes (used by loader)
# ─────────────────────────────────────────────

PLUGIN_TYPE_MAP = {
    "retrieval":  RetrievalPlugin,
    "tool":       ToolPlugin,
    "action":     ActionPlugin,
    "processing": ProcessingPlugin,
}

__all__ = [
    "RetrievalPlugin",
    "ToolPlugin",
    "ActionPlugin",
    "ProcessingPlugin",
    "PLUGIN_TYPE_MAP",
]
