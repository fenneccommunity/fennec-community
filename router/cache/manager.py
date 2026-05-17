"""
router_v2.cache.manager
~~~~~~~~~~~~~~~~~~~~~~~
A two-tier cache:

  Tier 1 — Route Decision Cache
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Stores (route_name, group_name) pairs keyed by a normalised query hash.
  Cache hits skip both the embedding model and the scoring pipeline —
  the cheapest path possible.

  Tier 2 — Embedding Cache
  ~~~~~~~~~~~~~~~~~~~~~~~~
  Stores query embeddings so that a repeated query only goes to the model
  once.  Managed inside EmbeddingProvider (see routing/pipeline.py);
  surfaced here for unified eviction and stats.

Eviction uses a combination of TTL (time-to-live) and LRU (least-recently-
used) order.  The LRU is maintained as an ``OrderedDict``.

Thread-safe.
"""
from __future__ import annotations

import collections
import hashlib
import threading
import time
from typing import Any, Dict, Optional, Tuple

from ..config import CacheConfig


# ---------------------------------------------------------------------------
# LRU + TTL Cache (generic)
# ---------------------------------------------------------------------------

class _LRUTTLCache:
    """
    Internal O(1) LRU cache with per-entry TTL.

    Keys and values must be simple Python objects (str, dict, list, …).
    """

    def __init__(self, max_size: int, ttl: int):
        self._max_size: int  = max_size
        self._ttl:      int  = ttl
        self._data:     Dict[str, Tuple[Any, float]] = collections.OrderedDict()
        self._lock:     threading.Lock = threading.Lock()
        self.hits:      int  = 0
        self.misses:    int  = 0

    # ---- Public interface -----------------------------------------------

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._data:
                self.misses += 1
                return None
            value, timestamp = self._data[key]
            if time.time() - timestamp > self._ttl:
                del self._data[key]
                self.misses += 1
                return None
            # Move to end (most recently used)
            self._data.move_to_end(key)
            self.hits += 1
            return value

    def set(self, key: str, value: Any):
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = (value, time.time())
            if len(self._data) > self._max_size:
                self._evict()

    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._data:
                del self._data[key]
                return True
            return False

    def clear(self):
        with self._lock:
            self._data.clear()
            self.hits   = 0
            self.misses = 0

    @property
    def size(self) -> int:
        return len(self._data)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total * 100 if total else 0.0

    # ---- Private --------------------------------------------------------

    def _evict(self):
        """Remove expired entries, then LRU entries until under max_size."""
        now = time.time()
        expired = [k for k, (_, ts) in self._data.items() if now - ts > self._ttl]
        for k in expired:
            del self._data[k]
        # Still over limit → drop LRU (front of OrderedDict)
        while len(self._data) > self._max_size:
            self._data.popitem(last=False)


# ---------------------------------------------------------------------------
# Cache Manager
# ---------------------------------------------------------------------------

class CacheManager:
    """
    Public interface for all caching in the router.

    Provides separate caches for:
    - **Route decisions** : maps query hash → (route_name, group_name)
    - **Group decisions** : maps query hash → group_name (group selection step)

    Usage
    -----
    ::

        cache = CacheManager(cfg)

        # Store a routing decision
        cache.set_decision(query, route_name="rag.docs_qa", group_name="rag")

        # Retrieve it on the next request
        hit = cache.get_decision(query)
        if hit:
            route_name, group_name = hit
    """

    def __init__(self, cfg: CacheConfig):
        self._cfg     = cfg
        self._enabled = cfg.enabled

        self._route_cache = _LRUTTLCache(
            max_size = cfg.max_size,
            ttl      = cfg.ttl_seconds,
        )

    # ------------------------------------------------------------------ #
    # Route decision cache
    # ------------------------------------------------------------------ #

    @staticmethod
    def _query_key(query: str) -> str:
        normalised = " ".join(query.lower().split())
        return hashlib.sha256(normalised.encode()).hexdigest()

    def get_decision(self, query: str) -> Optional[Tuple[str, Optional[str]]]:
        """
        Look up a cached routing decision.

        Returns
        -------
        (route_name, group_name) or None if not cached / expired.
        """
        if not self._enabled:
            return None
        return self._route_cache.get(self._query_key(query))

    def set_decision(
        self,
        query:      str,
        route_name: str,
        group_name: Optional[str] = None,
    ):
        """Cache a routing decision."""
        if not self._enabled:
            return
        self._route_cache.set(
            self._query_key(query),
            (route_name, group_name),
        )

    def invalidate(self, query: str) -> bool:
        """Remove a specific query from the cache (e.g. after route update)."""
        return self._route_cache.delete(self._query_key(query))

    def clear(self):
        """Flush all caches."""
        self._route_cache.clear()

    # ------------------------------------------------------------------ #
    # Stats
    # ------------------------------------------------------------------ #

    def stats(self) -> Dict[str, Any]:
        return {
            "enabled":        self._enabled,
            "ttl_seconds":    self._cfg.ttl_seconds,
            "route_cache": {
                "size":     self._route_cache.size,
                "max_size": self._cfg.max_size,
                "hits":     self._route_cache.hits,
                "misses":   self._route_cache.misses,
                "hit_rate": round(self._route_cache.hit_rate, 2),
            },
        }

    # ------------------------------------------------------------------ #
    # Python protocol
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"CacheManager(enabled={self._enabled}, "
            f"route_cache_size={self._route_cache.size})"
        )
