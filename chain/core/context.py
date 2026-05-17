"""
===============
Shared execution context that flows through the entire chain pipeline.

The ChainContext is a mutable bag that every chain in a pipeline can
read from and write to, enabling rich cross-chain communication without
tight coupling.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ChainInput:
    """
    Typed wrapper for data entering a chain.

    Attributes
    ----------
    data     : The primary payload passed between chains.
    metadata : Arbitrary key-value pairs attached by the caller.
    context  : Shared mutable bag (ChainContext or plain dict).
    timestamp: Creation time (epoch seconds).

    Example
    -------
    >>> inp = ChainInput(data="hello", metadata={"lang": "en"})
    >>> inp.context["user_intent"] = "greet"
    """

    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ChainOutput:
    """
    Typed wrapper for data leaving a chain.
    Attributes
    ----------
    data           : The result produced by the chain.
    metadata       : Enriched metadata from the chain.
    execution_time : Wall-clock duration in seconds.
    success        : Whether the chain completed without errors.
    error          : Human-readable error message when success=False.
    chain_id       : ID of the chain that produced this output.
    """

    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    success: bool = True
    error: Optional[str] = None
    chain_id: Optional[str] = None


class ChainContext:
    """
    Mutable shared context propagated across all chains in a pipeline.

    Think of it as a whiteboard every chain can read and write.

    Example
    -------
    >>> ctx = ChainContext(session_id="abc")
    >>> ctx["user_intent"] = "purchase"
    >>> ctx.set("cart_total", 99.9, namespace="shop")
    >>> ctx.get("cart_total", namespace="shop")
    99.9
    """

    def __init__(self, session_id: Optional[str] = None, **initial_values: Any):
        self._store: Dict[str, Any] = dict(initial_values)
        self.session_id: str = session_id or str(uuid.uuid4())
        self.created_at: float = time.time()
        self._namespaces: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # dict-like interface
    # ------------------------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        return self._store[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._store[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._store

    def get(self, key: str, default: Any = None, *, namespace: Optional[str] = None) -> Any:
        """Retrieve a value, optionally from a named namespace."""
        if namespace:
            return self._namespaces.get(namespace, {}).get(key, default)
        return self._store.get(key, default)

    def set(self, key: str, value: Any, *, namespace: Optional[str] = None) -> None:
        """Store a value, optionally under a named namespace."""
        if namespace:
            self._namespaces.setdefault(namespace, {})[key] = value
        else:
            self._store[key] = value

    def update(self, mapping: Dict[str, Any]) -> None:
        """Bulk update the default namespace."""
        self._store.update(mapping)

    def as_dict(self) -> Dict[str, Any]:
        """Return a shallow copy of the default namespace."""
        return dict(self._store)

    def __repr__(self) -> str:
        return f"ChainContext(session_id={self.session_id!r}, keys={list(self._store)})"
