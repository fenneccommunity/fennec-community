"""
Plugin Registry
===============
Centralized, searchable, AI-queryable plugin catalog.

  registry.register(plugin)
  registry.get("web_search")
  registry.search("find documents about climate")
  registry.list_by_type("retrieval")
  registry.to_tool_descriptors()   # → OpenAI function-calling list
"""

from __future__ import annotations
import logging
import re
from typing import Any, Dict, List, Optional, Type
from .base_plugin import BasePlugin
from .metadata import PluginMetadata
logger = logging.getLogger(__name__)


class PluginRegistry:
    """
    Single source of truth for all registered plugins.

    Thread-safety: designed for cooperative async; use a dedicated
    asyncio.Lock in the manager for concurrent registrations.
    """

    def __init__(self) -> None:
        self._plugins:    Dict[str, BasePlugin]          = {}
        self._classes:    Dict[str, Type[BasePlugin]]    = {}
        self._type_index: Dict[str, List[str]]           = {}   # type → [names]
        self._tag_index:  Dict[str, List[str]]           = {}   # tag → [names]

    # ─────────────────────────────────────────
    # Registration
    # ─────────────────────────────────────────

    def register(self, plugin: BasePlugin) -> None:
        name = plugin.name
        if name in self._plugins:
            raise KeyError(f"Plugin '{name}' is already registered.")
        self._plugins[name]  = plugin
        self._classes[name]  = type(plugin)
        self._index_plugin(plugin)
        logger.info("[Registry] Registered plugin: %s v%s", name, plugin.metadata.version)

    def unregister(self, name: str) -> None:
        if name not in self._plugins:
            raise KeyError(f"Plugin '{name}' is not registered.")
        plugin = self._plugins.pop(name)
        self._classes.pop(name, None)
        self._deindex_plugin(plugin)
        logger.info("[Registry] Unregistered plugin: %s", name)

    # ─────────────────────────────────────────
    # Lookup
    # ─────────────────────────────────────────

    def get(self, name: str) -> Optional[BasePlugin]:
        return self._plugins.get(name)

    def get_or_raise(self, name: str) -> BasePlugin:
        plugin = self._plugins.get(name)
        if plugin is None:
            raise KeyError(f"Plugin '{name}' not found in registry.")
        return plugin

    def exists(self, name: str) -> bool:
        return name in self._plugins

    def all(self) -> List[BasePlugin]:
        return list(self._plugins.values())

    def names(self) -> List[str]:
        return list(self._plugins.keys())

    # ─────────────────────────────────────────
    # Filtering
    # ─────────────────────────────────────────

    def list_by_type(self, plugin_type: str) -> List[BasePlugin]:
        """Return all plugins of a given PLUGIN_TYPE string."""
        names = self._type_index.get(plugin_type, [])
        return [self._plugins[n] for n in names if n in self._plugins]

    def list_by_tag(self, tag: str) -> List[BasePlugin]:
        names = self._tag_index.get(tag.lower(), [])
        return [self._plugins[n] for n in names if n in self._plugins]

    def list_enabled(self) -> List[BasePlugin]:
        return [p for p in self._plugins.values() if p.config.enabled]

    # ─────────────────────────────────────────
    # Keyword / capability search
    # ─────────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> List[BasePlugin]:
        """
        Simple keyword-based search over name, description, tags,
        capabilities, and use_cases.

        Returns up to top_k plugins sorted by relevance score.
        """
        tokens = set(re.findall(r"\w+", query.lower()))
        scored: List[tuple[float, BasePlugin]] = []

        for plugin in self._plugins.values():
            score = _keyword_score(tokens, plugin.metadata)
            if score > 0:
                scored.append((score, plugin))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:top_k]]

    def find_by_capability(self, capability: str) -> List[BasePlugin]:
        """Return plugins that explicitly list this capability."""
        cap = capability.lower()
        return [
            p for p in self._plugins.values()
            if any(cap in c.lower() for c in p.metadata.capabilities)
        ]

    # ─────────────────────────────────────────
    # AI / LLM integration
    # ─────────────────────────────────────────

    def to_tool_descriptors(
        self, plugin_names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Return OpenAI-format function-calling descriptors for all
        (or a subset of) registered plugins.
        """
        plugins = (
            [self._plugins[n] for n in plugin_names if n in self._plugins]
            if plugin_names
            else list(self._plugins.values())
        )
        return [p.metadata.to_tool_descriptor() for p in plugins if p.config.enabled]

    def to_anthropic_tools(
        self, plugin_names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Return Anthropic Claude tool-use descriptors."""
        plugins = (
            [self._plugins[n] for n in plugin_names if n in self._plugins]
            if plugin_names
            else list(self._plugins.values())
        )
        return [p.metadata.to_anthropic_tool() for p in plugins if p.config.enabled]

    # ─────────────────────────────────────────
    # Snapshot / export
    # ─────────────────────────────────────────

    def snapshot(self) -> List[Dict[str, Any]]:
        """Return a serializable summary of all registered plugins."""
        return [
            {
                "name":     p.name,
                "version":  p.metadata.version,
                "type":     getattr(type(p), "PLUGIN_TYPE", "unknown"),
                "status":   p.status.value,
                "enabled":  p.config.enabled,
                "tags":     p.metadata.tags,
                "capabilities": p.metadata.capabilities,
            }
            for p in self._plugins.values()
        ]

    def __len__(self) -> int:
        return len(self._plugins)

    def __contains__(self, name: str) -> bool:
        return name in self._plugins

    def __repr__(self) -> str:
        return f"PluginRegistry({len(self._plugins)} plugins: {list(self._plugins)})"

    # ─────────────────────────────────────────
    # Internal indexing
    # ─────────────────────────────────────────

    def _index_plugin(self, plugin: BasePlugin) -> None:
        ptype = getattr(type(plugin), "PLUGIN_TYPE", "unknown")
        self._type_index.setdefault(ptype, []).append(plugin.name)
        for tag in plugin.metadata.tags:
            self._tag_index.setdefault(tag.lower(), []).append(plugin.name)

    def _deindex_plugin(self, plugin: BasePlugin) -> None:
        ptype = getattr(type(plugin), "PLUGIN_TYPE", "unknown")
        if ptype in self._type_index:
            self._type_index[ptype] = [
                n for n in self._type_index[ptype] if n != plugin.name
            ]
        for tag in plugin.metadata.tags:
            key = tag.lower()
            if key in self._tag_index:
                self._tag_index[key] = [
                    n for n in self._tag_index[key] if n != plugin.name
                ]


# ─────────────────────────────────────────────
# Scoring helper
# ─────────────────────────────────────────────

def _keyword_score(tokens: set, meta: PluginMetadata) -> float:
    score = 0.0

    def _hit(text: str) -> float:
        words = set(re.findall(r"\w+", text.lower()))
        return len(tokens & words) / max(len(tokens), 1)

    score += _hit(meta.name)         * 3.0
    score += _hit(meta.description)  * 2.0
    for tag in meta.tags:
        score += _hit(tag)           * 2.5
    for cap in meta.capabilities:
        score += _hit(cap)           * 2.0
    for uc in meta.use_cases:
        score += _hit(uc)            * 1.5
    return score
