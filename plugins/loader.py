"""
Plugin Loader — Auto-Discovery & Dynamic Loading
================================================
Scans directories, imports plugin classes, and hands them to the registry.
Supports lazy loading (import-on-first-use) and hot-reload via file-mtime tracking.
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Type
from .base_plugin import BasePlugin, PluginConfig
from .registry import PluginRegistry

logger = logging.getLogger(__name__)


class PluginLoader:
    """
    Discovers and loads plugin classes from one or more directories.

    Usage
    -----
    loader = PluginLoader(registry, plugin_dirs=["./plugins/builtin", "./plugins/custom"])
    discovered = await loader.discover_all()

    Lazy loading
    ------------
    loader.register_lazy("my_plugin", "/path/to/my_plugin.py")
    plugin = loader.load_lazy("my_plugin")
    """

    def __init__(
        self,
        registry: PluginRegistry,
        plugin_dirs: Optional[List[str]] = None,
        default_config: Optional[PluginConfig] = None,
        module_prefix: str = "rag_plugins",
    ) -> None:
        self.registry       = registry
        self.plugin_dirs    = [Path(d) for d in (plugin_dirs or [])]
        self.default_config = default_config or PluginConfig()
        self.module_prefix  = module_prefix

        # name → Path  (lazy registry)
        self._lazy_map:    Dict[str, Path] = {}

        # file_path → mtime  (hot-reload tracking)
        self._file_mtimes: Dict[Path, float] = {}

        # names already loaded this session
        self._loaded_names: Set[str] = set()

    # ─────────────────────────────────────────
    # Primary API
    # ─────────────────────────────────────────

    async def discover_all(self) -> List[str]:
        """
        Scan all configured directories, load every valid plugin class found,
        and register it. Returns list of successfully registered plugin names.
        """
        registered: List[str] = []
        for d in self.plugin_dirs:
            found = await self._scan_directory(d)
            registered.extend(found)
        logger.info("[Loader] Discovery complete: %d plugin(s) registered.", len(registered))
        return registered

    async def discover_directory(self, directory: str) -> List[str]:
        """Discover plugins from a specific directory (not in self.plugin_dirs)."""
        return await self._scan_directory(Path(directory))

    def register_lazy(self, name: str, file_path: str) -> None:
        """
        Register a plugin for lazy loading.
        The class is NOT imported until load_lazy(name) is called.
        """
        self._lazy_map[name] = Path(file_path)
        logger.debug("[Loader] Lazy entry registered: %s → %s", name, file_path)

    def load_lazy(self, name: str) -> Optional[BasePlugin]:
        """
        Import and instantiate a lazily-registered plugin on demand.
        Returns the plugin instance (already added to registry) or None.
        """
        if name not in self._lazy_map:
            logger.warning("[Loader] No lazy entry for '%s'.", name)
            return None
        path = self._lazy_map[name]
        instances = self._load_file(path)
        for instance in instances:
            if instance.name == name:
                return instance
        return None

    # ─────────────────────────────────────────
    # Hot-reload support
    # ─────────────────────────────────────────

    def get_changed_files(self) -> List[Path]:
        """Return list of plugin files that have changed since last load."""
        changed: List[Path] = []
        for path, mtime in self._file_mtimes.items():
            if path.exists() and path.stat().st_mtime > mtime:
                changed.append(path)
        return changed

    async def reload_file(self, file_path: Path) -> List[str]:
        """
        Reload plugins from a changed file:
          1. Unregister old versions.
          2. Re-import and re-register.
        """
        reloaded: List[str] = []

        # Unregister old instances from this file
        for name in list(self._loaded_names):
            plugin = self.registry.get(name)
            if plugin is None:
                continue
            src = getattr(type(plugin), "__module__", "")
            if file_path.stem in src:
                try:
                    self.registry.unregister(name)
                    self._loaded_names.discard(name)
                except Exception as exc:
                    logger.warning("[Loader] Failed to unregister %s: %s", name, exc)

        # Remove cached module so Python re-executes the file
        module_name = f"{self.module_prefix}.{file_path.stem}"
        sys.modules.pop(module_name, None)

        # Re-load
        instances = self._load_file(file_path)
        for inst in instances:
            reloaded.append(inst.name)

        logger.info("[Loader] Reloaded %d plugin(s) from %s.", len(reloaded), file_path.name)
        return reloaded

    # ─────────────────────────────────────────
    # Internals
    # ─────────────────────────────────────────

    async def _scan_directory(self, directory: Path) -> List[str]:
        if not directory.exists():
            logger.warning("[Loader] Directory not found: %s", directory)
            return []

        registered: List[str] = []
        for py_file in sorted(directory.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            instances = self._load_file(py_file)
            for inst in instances:
                registered.append(inst.name)
        return registered

    def _load_file(self, path: Path) -> List[BasePlugin]:
        """Import a .py file and register all BasePlugin subclasses found."""
        module_name = f"{self.module_prefix}.{path.stem}"
        instances: List[BasePlugin] = []

        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                logger.error("[Loader] Cannot create spec for %s", path)
                return instances

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)  # type: ignore[attr-defined]

            self._file_mtimes[path] = path.stat().st_mtime

            plugin_classes = _find_plugin_classes(module)
            for cls in plugin_classes:
                inst = self._instantiate(cls, path)
                if inst is not None:
                    instances.append(inst)

        except Exception as exc:
            logger.error("[Loader] Failed to load %s: %s", path, exc, exc_info=True)

        return instances

    def _instantiate(self, cls: Type[BasePlugin], source: Path) -> Optional[BasePlugin]:
        name = getattr(getattr(cls, "METADATA", None), "name", cls.__name__)
        try:
            if self.registry.exists(name):
                logger.debug("[Loader] Skipping already-registered plugin: %s", name)
                return None
            instance = cls(config=self.default_config)
            self.registry.register(instance)
            self._loaded_names.add(name)
            logger.info("[Loader] Loaded plugin '%s' from %s", name, source.name)
            return instance
        except Exception as exc:
            logger.error(
                "[Loader] Could not instantiate %s from %s: %s",
                cls.__name__, source.name, exc, exc_info=True,
            )
            return None


# ─────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────

def _find_plugin_classes(module) -> List[Type[BasePlugin]]:
    """Return all concrete BasePlugin subclasses defined in a module."""
    found: List[Type[BasePlugin]] = []
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if (
            issubclass(obj, BasePlugin)
            and obj is not BasePlugin
            and obj.__module__ == module.__name__
            and not inspect.isabstract(obj)
            and hasattr(obj, "METADATA")
        ):
            found.append(obj)
    return found
