"""
Plugin Manager — Production-Grade Orchestrator
==============================================
Central façade that coordinates:
  • Registry       — catalog of all plugins
  • Loader         — auto-discovery & hot-reload
  • Security       — permissions, sandboxing, validation
  • AI Selector    — smart plugin routing for agents
  • Lifecycle      — initialize / execute / cleanup / shutdown
  • Observability  — structured metrics, hooks, health dashboard
"""

from __future__ import annotations
import asyncio
import logging
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple
from .base_plugin import BasePlugin, PluginConfig, PluginError, PluginTimeoutError
from .loader import PluginLoader
from .metadata import CostTier, ExecutionContext, PluginStatus
from .registry import PluginRegistry
from .security import ExecutionSandbox, InputSanitizer, PermissionGuard
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Observability dataclasses
# ─────────────────────────────────────────────

class PluginObservability:
    """Lightweight in-process metrics store."""

    def __init__(self) -> None:
        self._execs: Dict[str, List[Dict]] = defaultdict(list)
        self._errors: List[Dict]           = []
        self._hook_calls: Dict[str, int]   = defaultdict(int)

    def record_exec(
        self, name: str, elapsed: float, success: bool, error: str = ""
    ) -> None:
        self._execs[name].append({
            "ts":      time.time(),
            "elapsed": elapsed,
            "success": success,
            "error":   error,
        })
        # Keep only last 200 per plugin
        if len(self._execs[name]) > 200:
            self._execs[name].pop(0)
        if not success:
            self._errors.append({"plugin": name, "ts": time.time(), "msg": error})
            if len(self._errors) > 500:
                self._errors.pop(0)

    def record_hook(self, event: str) -> None:
        self._hook_calls[event] += 1

    def summary(self) -> Dict[str, Any]:
        total = sum(len(v) for v in self._execs.values())
        failed = sum(
            1 for records in self._execs.values()
            for r in records if not r["success"]
        )
        top_plugins = sorted(
            {k: len(v) for k, v in self._execs.items()}.items(),
            key=lambda x: x[1], reverse=True,
        )[:5]
        return {
            "total_executions":    total,
            "total_failures":      failed,
            "error_rate_pct":      round(failed / total * 100, 1) if total else 0,
            "top_plugins":         top_plugins,
            "hook_call_counts":    dict(self._hook_calls),
            "recent_errors":       self._errors[-10:],
        }

    def plugin_report(self, name: str) -> Dict[str, Any]:
        records = self._execs.get(name, [])
        if not records:
            return {"name": name, "executions": 0}
        elapsed = [r["elapsed"] for r in records]
        successes = [r for r in records if r["success"]]
        return {
            "name":         name,
            "executions":   len(records),
            "success_rate": round(len(successes) / len(records) * 100, 1),
            "avg_ms":       round(sum(elapsed) / len(elapsed) * 1000, 1),
            "max_ms":       round(max(elapsed) * 1000, 1),
            "min_ms":       round(min(elapsed) * 1000, 1),
        }


# ─────────────────────────────────────────────
# AI Selector
# ─────────────────────────────────────────────

class AIPluginSelector:
    """
    Rule-based + keyword-semantic plugin selector.

    For full semantic search, inject an embed_fn (str → List[float]).
    Without it, falls back to keyword scoring from the registry.
    """

    def __init__(
        self,
        registry: PluginRegistry,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
    ) -> None:
        self.registry = registry
        self.embed_fn = embed_fn

    def select(
        self,
        query:       str,
        plugin_type: Optional[str] = None,
        max_cost:    Optional[CostTier] = None,
        top_k:       int = 3,
    ) -> List[BasePlugin]:
        """
        Select the best plugins for a query.

        Parameters
        ----------
        query       : natural-language description of what is needed
        plugin_type : restrict to a plugin type ("retrieval", "tool", …)
        max_cost    : filter out plugins more expensive than this tier
        top_k       : maximum number of suggestions
        """
        candidates: List[BasePlugin] = (
            self.registry.list_by_type(plugin_type)
            if plugin_type
            else self.registry.list_enabled()
        )

        # Cost filter
        tier_rank = {t: i for i, t in enumerate(CostTier)}
        if max_cost is not None:
            max_rank = tier_rank[max_cost]
            candidates = [
                p for p in candidates
                if tier_rank.get(p.metadata.cost_tier, 0) <= max_rank
            ]

        if not candidates:
            return []

        # Keyword scoring (always available)
        from .registry import _keyword_score
        import re
        tokens = set(re.findall(r"\w+", query.lower()))
        scored: List[Tuple[float, BasePlugin]] = [
            (_keyword_score(tokens, p.metadata), p)
            for p in candidates
        ]

        # Semantic re-ranking (optional)
        if self.embed_fn is not None:
            scored = self._semantic_rerank(query, scored)

        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:top_k]]

    def select_one(self, query: str, **kwargs) -> Optional[BasePlugin]:
        results = self.select(query, top_k=1, **kwargs)
        return results[0] if results else None

    def _semantic_rerank(
        self,
        query: str,
        scored: List[Tuple[float, BasePlugin]],
    ) -> List[Tuple[float, BasePlugin]]:
        """Combine keyword score with cosine similarity."""
        import math
        try:
            q_vec = self.embed_fn(query)
            reranked: List[Tuple[float, BasePlugin]] = []
            for kw_score, plugin in scored:
                doc = " ".join([
                    plugin.metadata.description,
                    *plugin.metadata.use_cases,
                    *plugin.metadata.capabilities,
                ])
                d_vec = self.embed_fn(doc)
                cos   = _cosine(q_vec, d_vec)
                reranked.append((kw_score * 0.4 + cos * 0.6, plugin))
            return reranked
        except Exception as exc:
            logger.warning("[AISelector] Semantic rerank failed, using keyword only: %s", exc)
            return scored


def _cosine(a: List[float], b: List[float]) -> float:
    import math
    dot  = sum(x * y for x, y in zip(a, b))
    na   = math.sqrt(sum(x * x for x in a))
    nb   = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-9)


# ─────────────────────────────────────────────
# Plugin Manager
# ─────────────────────────────────────────────

class PluginManager:
    """
    Production-grade plugin manager.

    Wires together registry, loader, security, selector, and observability
    into a single coherent API.

    Quick-start
    -----------
    manager = PluginManager(plugin_dirs=["./my_plugins"])
    await manager.startup()

    # Execute by name
    result = await manager.execute("web_search", {"query": "RAG"}, context)

    # Let an AI agent pick the best plugin
    plugin  = manager.selector.select_one("search for recent papers on LLMs")
    result  = await manager.execute(plugin.name, {...}, context)

    # Get LLM-ready tool descriptors
    tools   = manager.registry.to_tool_descriptors()
    """

    def __init__(
        self,
        plugin_dirs:     Optional[List[str]] = None,
        system_version:  str                 = "1.0.0",
        safe_mode:       bool                = False,
        auto_discover:   bool                = True,
        default_config:  Optional[PluginConfig] = None,
        embed_fn:        Optional[Callable[[str], List[float]]] = None,
        hard_timeout:    float               = 60.0,
    ) -> None:
        self.system_version  = system_version
        self.safe_mode        = safe_mode
        self.auto_discover    = auto_discover
        self._shutting_down   = False

        # Core subsystems
        self.registry    = PluginRegistry()
        self.loader      = PluginLoader(
            self.registry,
            plugin_dirs=plugin_dirs or [],
            default_config=default_config,
        )
        self.security    = PermissionGuard()
        self.sandbox     = ExecutionSandbox(hard_timeout_sec=hard_timeout)
        self.selector    = AIPluginSelector(self.registry, embed_fn=embed_fn)
        self.metrics     = PluginObservability()

        # Hook system: event → [callbacks]
        self._hooks: Dict[str, List[Tuple[int, Callable]]] = defaultdict(list)

        # Hot-reload state
        self._hot_reload_task: Optional[asyncio.Task] = None
        self._hot_reload_on   = False

        logger.info(
            "[PluginManager] Initialized (system=%s, safe_mode=%s, auto_discover=%s)",
            system_version, safe_mode, auto_discover,
        )

    # ─────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────

    async def startup(self) -> None:
        """Initialize the manager: auto-discover plugins and fire startup hooks."""
        logger.info("[PluginManager] Starting up...")
        if self.auto_discover:
            found = await self.loader.discover_all()
            logger.info("[PluginManager] Auto-discovered %d plugin(s).", len(found))

            # Grant required permissions in non-safe mode
            if not self.safe_mode:
                for name in found:
                    plugin = self.registry.get(name)
                    if plugin:
                        self.security.grant_all_required(plugin)

        # Initialize all registered plugins
        for plugin in self.registry.all():
            await self._initialize_plugin(plugin)

        await self._fire_hook("system.startup", manager=self)
        logger.info("[PluginManager] Startup complete. %d plugin(s) active.", len(self.registry))

    async def shutdown(self) -> None:
        """Gracefully shutdown: cleanup all plugins and stop background tasks."""
        logger.info("[PluginManager] Shutting down...")
        self._shutting_down = True

        await self.disable_hot_reload()
        await self._fire_hook("system.shutdown", manager=self)

        for plugin in self.registry.all():
            try:
                await plugin.cleanup()
                plugin.status = PluginStatus.DISABLED
            except Exception as exc:
                logger.warning("[PluginManager] Cleanup error for %s: %s", plugin.name, exc)

        logger.info("[PluginManager] Shutdown complete.")

    # ─────────────────────────────────────────
    # Manual plugin registration
    # ─────────────────────────────────────────

    async def register(self, plugin: BasePlugin) -> bool:
        """
        Manually register a plugin instance.
        Runs compatibility check → permission check → initialize().
        """
        # Compatibility
        if not plugin.metadata.is_compatible_with(self.system_version):
            logger.error(
                "[PluginManager] [%s] Incompatible with system version %s.",
                plugin.name, self.system_version,
            )
            return False

        # Check dependency plugins are already registered
        missing_deps = [
            d for d in plugin.metadata.dependencies
            if not self.registry.exists(d)
        ]
        if missing_deps:
            logger.error(
                "[PluginManager] [%s] Missing dependencies: %s",
                plugin.name, missing_deps,
            )
            return False

        # Permission check in safe mode
        if self.safe_mode:
            ok = self.security.check_all_required(plugin)
            if not ok:
                logger.error(
                    "[PluginManager] [%s] Permission check failed (safe_mode=True).",
                    plugin.name,
                )
                return False

        try:
            self.registry.register(plugin)
        except KeyError as exc:
            logger.warning("[PluginManager] %s", exc)
            return False

        ok = await self._initialize_plugin(plugin)
        if ok:
            await self._fire_hook("plugin.registered", plugin=plugin)
        return ok

    async def unregister(self, name: str, force: bool = False) -> bool:
        """Unregister and cleanup a plugin."""
        plugin = self.registry.get(name)
        if plugin is None:
            logger.warning("[PluginManager] Cannot unregister '%s': not found.", name)
            return False

        # Check nothing depends on this plugin
        if not force:
            dependents = [
                p.name for p in self.registry.all()
                if name in p.metadata.dependencies
            ]
            if dependents:
                logger.error(
                    "[PluginManager] Cannot unregister '%s': depended on by %s",
                    name, dependents,
                )
                return False

        try:
            await plugin.cleanup()
            self.registry.unregister(name)
            await self._fire_hook("plugin.unregistered", plugin_name=name)
            return True
        except Exception as exc:
            logger.error("[PluginManager] Unregister error for '%s': %s", name, exc)
            return False

    # ─────────────────────────────────────────
    # Execution
    # ─────────────────────────────────────────

    async def execute(
        self,
        name:       str,
        input_data: Dict[str, Any],
        context:    Optional[ExecutionContext] = None,
        use_retry:  bool = False,
    ) -> Any:
        """
        Execute a plugin by name.

        Parameters
        ----------
        name       : plugin name
        input_data : raw input dict (validated by plugin + sanitizer)
        context    : runtime context (query, session, memory handles, …)
        use_retry  : enable exponential-backoff retry
        """
        if self._shutting_down:
            raise PluginError("PluginManager is shutting down.")

        plugin = self.registry.get_or_raise(name)
        ctx    = context or ExecutionContext()

        if not plugin.config.enabled:
            raise PluginError(f"[{name}] Plugin is disabled.")

        # Security: permission enforcement
        if self.safe_mode:
            self.security.check_all_required(plugin, ctx)

        # Sanitize inputs against schema
        if plugin.metadata.input_schema:
            input_data = InputSanitizer.sanitize(
                input_data,
                plugin.metadata.input_schema,
                strict=False,   # lenient: ignore unknown keys
            )

        t0 = time.monotonic()
        try:
            if use_retry:
                result = await plugin.safe_execute_with_retry(input_data, ctx)
            else:
                result = await plugin.safe_execute(input_data, ctx)

            elapsed = time.monotonic() - t0
            self.metrics.record_exec(name, elapsed, True)
            await self._fire_hook("plugin.executed", plugin=plugin, result=result)
            return result

        except (PluginError, PluginTimeoutError) as exc:
            elapsed = time.monotonic() - t0
            self.metrics.record_exec(name, elapsed, False, str(exc))
            await self._fire_hook("plugin.failed", plugin=plugin, error=str(exc))
            raise

    async def execute_batch(
        self,
        calls:     List[Tuple[str, Dict[str, Any]]],
        context:   Optional[ExecutionContext] = None,
        parallel:  bool = True,
    ) -> Dict[str, Any]:
        """
        Execute multiple plugins.

        Parameters
        ----------
        calls    : list of (plugin_name, input_data) pairs
        parallel : run concurrently (True) or sequentially (False)
        """
        results: Dict[str, Any] = {}

        if parallel:
            tasks = {
                name: self.execute(name, data, context)
                for name, data in calls
            }
            settled = await asyncio.gather(*tasks.values(), return_exceptions=True)
            for (name, _), outcome in zip(calls, settled):
                results[name] = (
                    {"error": str(outcome)} if isinstance(outcome, Exception)
                    else outcome
                )
        else:
            for name, data in calls:
                try:
                    results[name] = await self.execute(name, data, context)
                except Exception as exc:
                    results[name] = {"error": str(exc)}

        return results

    async def execute_pipeline(
        self,
        pipeline:  List[str],
        initial:   Dict[str, Any],
        context:   Optional[ExecutionContext] = None,
    ) -> Any:
        """
        Execute a sequential pipeline where the output of plugin[n]
        becomes the input to plugin[n+1].
        """
        data = initial
        ctx  = context or ExecutionContext()
        for name in pipeline:
            result = await self.execute(name, data if isinstance(data, dict) else {}, ctx)
            data   = result
        return data

    # ─────────────────────────────────────────
    # AI agent helpers
    # ─────────────────────────────────────────

    def select(
        self,
        query:       str,
        plugin_type: Optional[str]     = None,
        max_cost:    Optional[CostTier] = None,
        top_k:       int               = 3,
    ) -> List[BasePlugin]:
        """Shortcut to selector.select()."""
        return self.selector.select(query, plugin_type, max_cost, top_k)

    def select_one(self, query: str, **kwargs) -> Optional[BasePlugin]:
        """Return the single best plugin for the query."""
        return self.selector.select_one(query, **kwargs)

    async def auto_execute(
        self,
        query:      str,
        input_data: Optional[Dict[str, Any]] = None,
        context:    Optional[ExecutionContext] = None,
        **selector_kwargs,
    ) -> Optional[Any]:
        """
        Let the AI selector pick the best plugin and execute it.
        Returns None if no suitable plugin is found.
        """
        plugin = self.select_one(query, **selector_kwargs)
        if plugin is None:
            logger.warning("[PluginManager] No plugin selected for query: %s", query[:80])
            return None
        data = input_data or {"query": query}
        return await self.execute(plugin.name, data, context)

    def tool_descriptors(self) -> List[Dict[str, Any]]:
        """Return OpenAI function-calling descriptors for all enabled plugins."""
        return self.registry.to_tool_descriptors()

    def anthropic_tools(self) -> List[Dict[str, Any]]:
        """Return Anthropic tool-use descriptors for all enabled plugins."""
        return self.registry.to_anthropic_tools()

    # ─────────────────────────────────────────
    # Hot reload
    # ─────────────────────────────────────────

    async def enable_hot_reload(self, interval: float = 5.0) -> None:
        if self._hot_reload_on:
            return
        self._hot_reload_on = True
        self._hot_reload_task = asyncio.create_task(
            self._hot_reload_loop(interval)
        )
        logger.info("[PluginManager] Hot-reload enabled (interval=%.1fs).", interval)

    async def disable_hot_reload(self) -> None:
        self._hot_reload_on = False
        if self._hot_reload_task:
            self._hot_reload_task.cancel()
            try:
                await self._hot_reload_task
            except asyncio.CancelledError:
                pass
        logger.info("[PluginManager] Hot-reload disabled.")

    async def _hot_reload_loop(self, interval: float) -> None:
        while self._hot_reload_on and not self._shutting_down:
            try:
                await asyncio.sleep(interval)
                changed = self.loader.get_changed_files()
                for path in changed:
                    logger.info("[PluginManager] File changed: %s — reloading.", path.name)
                    await self.loader.reload_file(path)
                    # Re-initialize newly loaded plugins
                    for plugin in self.registry.all():
                        if plugin.status == PluginStatus.DISABLED:
                            await self._initialize_plugin(plugin)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("[PluginManager] Hot-reload error: %s", exc)

    # ─────────────────────────────────────────
    # Hook system
    # ─────────────────────────────────────────

    def on(self, event: str, callback: Callable, priority: int = 50) -> None:
        """Register a hook callback for an event."""
        if not callable(callback):
            raise TypeError("callback must be callable")
        self._hooks[event].append((priority, callback))
        self._hooks[event].sort(key=lambda x: x[0], reverse=True)

    def off(self, event: str, callback: Callable) -> None:
        """Remove a hook callback."""
        self._hooks[event] = [
            (p, cb) for p, cb in self._hooks[event] if cb is not callback
        ]

    async def _fire_hook(self, event: str, **kwargs) -> None:
        self.metrics.record_hook(event)
        for _, cb in self._hooks.get(event, []):
            try:
                import inspect
                if inspect.iscoroutinefunction(cb):
                    await cb(**kwargs)
                else:
                    cb(**kwargs)
            except Exception as exc:
                logger.warning("[PluginManager] Hook '%s' error: %s", event, exc)

    # ─────────────────────────────────────────
    # Observability
    # ─────────────────────────────────────────

    def dashboard(self) -> Dict[str, Any]:
        """Full health + metrics snapshot of the plugin ecosystem."""
        return {
            "system_version":    self.system_version,
            "safe_mode":         self.safe_mode,
            "hot_reload":        self._hot_reload_on,
            "total_plugins":     len(self.registry),
            "enabled_plugins":   len(self.registry.list_enabled()),
            "plugins":           [
                {**p.get_stats(), **{"health": p.get_health()}}
                for p in self.registry.all()
            ],
            "metrics":           self.metrics.summary(),
        }

    def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        return {p.name: p.get_health() for p in self.registry.all()}

    # ─────────────────────────────────────────
    # Convenience accessors
    # ─────────────────────────────────────────

    def get(self, name: str) -> Optional[BasePlugin]:
        return self.registry.get(name)

    def list_plugins(self, plugin_type: Optional[str] = None) -> List[Dict[str, Any]]:
        plugins = (
            self.registry.list_by_type(plugin_type)
            if plugin_type
            else self.registry.all()
        )
        return [
            {
                "name":         p.name,
                "version":      p.metadata.version,
                "type":         getattr(type(p), "PLUGIN_TYPE", "unknown"),
                "enabled":      p.config.enabled,
                "status":       p.status.value,
                "capabilities": p.metadata.capabilities,
                "tags":         p.metadata.tags,
                "cost_tier":    p.metadata.cost_tier.value,
            }
            for p in plugins
        ]

    def grant_permission(self, plugin_name: str, permission: str) -> None:
        self.security.grant(plugin_name, permission)

    def revoke_permission(self, plugin_name: str, permission: str) -> None:
        self.security.revoke(plugin_name, permission)

    # ─────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────

    async def _initialize_plugin(self, plugin: BasePlugin) -> bool:
        plugin.status = PluginStatus.INITIALIZING
        try:
            ok = await plugin.initialize()
            if ok:
                plugin.status = PluginStatus.ENABLED
                plugin.initialized_at = __import__("datetime").datetime.utcnow()
                logger.info("[PluginManager] Initialized: %s", plugin.name)
                return True
            else:
                plugin.status    = PluginStatus.ERROR
                plugin.error_msg = "initialize() returned False"
                logger.error("[PluginManager] Init failed for: %s", plugin.name)
                return False
        except Exception as exc:
            plugin.status    = PluginStatus.ERROR
            plugin.error_msg = str(exc)
            logger.error("[PluginManager] Init error for %s: %s", plugin.name, exc)
            return False

    def __repr__(self) -> str:
        return (
            f"PluginManager(plugins={len(self.registry)}, "
            f"system={self.system_version}, safe_mode={self.safe_mode})"
        )
