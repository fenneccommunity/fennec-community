"""
================
Utility helpers:
- ``cached`` — function-level result cache decorator
- ``CachingChain`` — wrap any chain with a result cache
- ``DeclarativeChainBuilder`` — build pipelines from YAML / JSON config
- ``build_from_dict`` / ``build_from_yaml`` / ``build_from_json``


YAML example
------------
steps:
  - type: transform
    name: strip_whitespace
  - type: transform
    name: to_upper
  - type: parallel
    branches:
      summary: transform
      length: transform
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
from typing import Any, Callable, Dict, Optional, Type

from ..core.base import BaseChain
from ..core.config import ChainConfig
from ..core.context import ChainInput, ChainOutput
from ..core.tracing import ExecutionTracer

logger = logging.getLogger(__name__)


# ============================================================================
# Simple in-memory cache decorator
# ============================================================================


def cached(fn: Callable) -> Callable:
    """
    LRU-style memoisation for synchronous callables.

    Uses ``functools.lru_cache`` under the hood but handles unhashable
    arguments gracefully by falling back to a JSON-keyed dict cache.
    """
    _store: Dict[str, Any] = {}

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            raw_key = json.dumps({"a": args, "k": kwargs}, sort_keys=True, default=str)
            key = hashlib.sha256(raw_key.encode()).hexdigest()
        except Exception:
            return fn(*args, **kwargs)

        if key not in _store:
            _store[key] = fn(*args, **kwargs)
        return _store[key]

    wrapper.cache_clear = _store.clear  # type: ignore[attr-defined]
    return wrapper


# ============================================================================
# CachingChain
# ============================================================================


class CachingChain(BaseChain):
    """
    Wraps another chain and caches its results in memory.

    The cache key is derived from a SHA-256 hash of the serialised input.
    Useful for expensive LLM calls that repeat the same inputs.

    Example
    -------
    >>> slow_chain = MyExpensiveChain()
    >>> chain = CachingChain(slow_chain)
    >>> await chain.arun("query")   # executes slow_chain
    >>> await chain.arun("query")   # returns from cache ⚡
    """

    def __init__(
        self,
        chain: BaseChain,
        name: Optional[str] = None,
        config: Optional[ChainConfig] = None,
        tracer: Optional[ExecutionTracer] = None,
    ) -> None:
        super().__init__(name=name or f"Cached({chain.name})", config=config, tracer=tracer)
        self._inner = chain
        self._cache: Dict[str, Any] = {}
        self.hit_count = 0
        self.miss_count = 0

    def _cache_key(self, data: Any) -> str:
        try:
            raw = json.dumps(data, sort_keys=True, default=str)
        except Exception:
            raw = str(data)
        return hashlib.sha256(raw.encode()).hexdigest()

    async def _aexecute(self, input_data: ChainInput) -> ChainOutput:
        key = self._cache_key(input_data.data)
        if key in self._cache:
            self.hit_count += 1
            logger.debug("[%s] cache HIT  key=%s…", self.name, key[:12])
            return ChainOutput(
                data=self._cache[key],
                metadata={"cache": "hit"},
                success=True,
            )

        self.miss_count += 1
        logger.debug("[%s] cache MISS key=%s…", self.name, key[:12])
        self._inner.tracer = self.tracer
        out = await self._inner._aexecute(input_data)
        if out.success:
            self._cache[key] = out.data
        return out

    def cache_clear(self) -> None:
        self._cache.clear()
        self.hit_count = 0
        self.miss_count = 0

    @property
    def cache_stats(self) -> Dict[str, int]:
        return {"hits": self.hit_count, "misses": self.miss_count, "size": len(self._cache)}


# ============================================================================
# Declarative builder
# ============================================================================


class DeclarativeChainBuilder:
    """
    Build chain pipelines from a plain-dict / YAML / JSON specification.

    Spec format (dict)
    ------------------
    {
        "type": "sequential",   # or "parallel", "transform", etc.
        "name": "my_pipeline",  # optional
        "steps": [              # for sequential
            {"type": "transform", "fn": <callable>},
            {"type": "parallel",  "branches": {"a": {...}, "b": {...}}},
        ]
    }

    Example (JSON string)
    ----------------------
    >>> spec_json = '''
    ... {
    ...   "type": "sequential",
    ...   "steps": [
    ...     {"type": "transform", "fn_name": "strip"},
    ...     {"type": "transform", "fn_name": "upper"}
    ...   ]
    ... }
    ... '''
    >>> chain = DeclarativeChainBuilder.from_json(spec_json, fn_registry={"strip": str.strip, "upper": str.upper})
    >>> await chain.arun("  hello  ")
    'HELLO'
    """

    @staticmethod
    def from_dict(
        spec: Dict[str, Any],
        fn_registry: Optional[Dict[str, Callable]] = None,
        chain_registry: Optional[Dict[str, Type[BaseChain]]] = None,
    ) -> BaseChain:
        """Build from a plain dict."""
        from ..chains.sequential import SequentialChain
        from ..chains.parallel import ParallelChain
        from ..chains.conditional import TransformChain, ConditionalChain
        from ..registry.registry import ChainRegistry

        fn_registry = fn_registry or {}
        chain_registry = chain_registry or {}

        chain_type = spec.get("type", "sequential").lower()
        name = spec.get("name")

        if chain_type == "sequential":
            steps = [
                DeclarativeChainBuilder.from_dict(s, fn_registry, chain_registry)
                for s in spec.get("steps", [])
            ]
            return SequentialChain(steps, name=name or "pipeline")

        elif chain_type == "parallel":
            branches_spec = spec.get("branches", {})
            if isinstance(branches_spec, dict):
                branches = {
                    k: DeclarativeChainBuilder.from_dict(v, fn_registry, chain_registry)
                    for k, v in branches_spec.items()
                }
            else:
                branches = [
                    DeclarativeChainBuilder.from_dict(v, fn_registry, chain_registry)
                    for v in branches_spec
                ]
            return ParallelChain(branches, name=name or "parallel")

        elif chain_type == "transform":
            fn_name = spec.get("fn_name") or spec.get("fn")
            if callable(fn_name):
                fn = fn_name
            elif isinstance(fn_name, str) and fn_name in fn_registry:
                fn = fn_registry[fn_name]
            else:
                raise ValueError(f"Unknown fn '{fn_name}' for TransformChain")
            return TransformChain(fn, name=name or fn_name)

        elif chain_type in ChainRegistry._registry:
            cls = ChainRegistry.get(chain_type)
            return cls(name=name)

        else:
            raise ValueError(f"Unknown chain type: '{chain_type}'")

    @staticmethod
    def from_json(
        json_str: str,
        fn_registry: Optional[Dict[str, Callable]] = None,
        chain_registry: Optional[Dict[str, Type[BaseChain]]] = None,
    ) -> BaseChain:
        """Build from a JSON string."""
        spec = json.loads(json_str)
        return DeclarativeChainBuilder.from_dict(spec, fn_registry, chain_registry)

    @staticmethod
    def from_yaml(
        yaml_str: str,
        fn_registry: Optional[Dict[str, Callable]] = None,
        chain_registry: Optional[Dict[str, Type[BaseChain]]] = None,
    ) -> BaseChain:
        """
        Build from a YAML string.
        Requires ``pyyaml`` to be installed (``pip install pyyaml``).
        """
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("pyyaml is required for YAML support: pip install pyyaml") from exc
        spec = yaml.safe_load(yaml_str)
        return DeclarativeChainBuilder.from_dict(spec, fn_registry, chain_registry)
