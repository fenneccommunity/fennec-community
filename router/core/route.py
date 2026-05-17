"""

Enhanced Route that supports:
  - BaseHandler interface (while staying backward-compatible with callables)
  - Keyword signals for hybrid scoring
  - Dynamic score adjustment (feedback loop)
  - Tool chaining declarations
  - Full serialisation / deserialisation
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Union

from .base import BaseHandler, CallableHandler, HandlerRequest, HandlerResponse


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@dataclass
class RouteKeywords:
    """
    Keyword signals that improve routing accuracy without an embedding call.

    Fields
    ------
    required    : ALL of these keywords must appear in the query (AND logic).
    any_of      : At least one of these must appear (OR logic).
    excluded    : If any of these appear, this route is vetoed.
    boost       : +score bonus applied per matched keyword (0–1 scale).
    """
    required:   List[str] = field(default_factory=list)
    any_of:     List[str] = field(default_factory=list)
    excluded:   List[str] = field(default_factory=list)
    boost:      float     = 0.05    # Added to combined score per matched keyword


class Route:
    """
    A named routing destination with semantic examples, keyword signals,
    a handler, and runtime metrics.

    Accepts either a :class:`~router_v2.core.base.BaseHandler` instance
    **or** a plain callable (wrapped transparently via :class:`CallableHandler`).

    Parameters
    ----------
    name        : Unique identifier for the route.
    description : Human-readable description (also used for semantic scoring).
    handler     : A ``BaseHandler`` instance or a callable.
    examples    : Sample utterances for semantic matching.
    keywords    : Optional keyword-signal configuration.
    group       : Parent group name (set by RouteGroup).
    tags        : Free-form labels for filtering / introspection.
    metadata    : Arbitrary key-value store.
    enabled     : Whether the route participates in routing.
    priority    : Tie-breaker (higher wins).  Used after scoring.
    tools       : Names of tools this route may chain to.
    score_bias  : Persistent score offset updated by the feedback engine.
    """

    def __init__(
        self,
        name:        str,
        description: str,
        handler:     Union[BaseHandler, Callable],
        examples:    Optional[List[str]]        = None,
        keywords:    Optional[RouteKeywords]    = None,
        group:       Optional[str]              = None,
        tags:        Optional[Set[str]]         = None,
        metadata:    Optional[Dict[str, Any]]   = None,
        enabled:     bool                       = True,
        priority:    int                        = 0,
        tools:       Optional[List[str]]        = None,
        score_bias:  float                      = 0.0,
    ):
        if not name or not name.strip():
            raise ValueError("Route name cannot be empty.")
        if not description or not description.strip():
            raise ValueError("Route description cannot be empty.")

        self.name         = name.strip()
        self.description  = description.strip()
        self.handler: BaseHandler = (
            handler if isinstance(handler, BaseHandler)
            else CallableHandler(handler)
        )
        self.examples:  List[str]        = list(dict.fromkeys(examples or []))
        self.keywords:  RouteKeywords    = keywords or RouteKeywords()
        self.group:     Optional[str]    = group
        self.tags:      Set[str]         = tags or set()
        self.metadata:  Dict[str, Any]   = metadata or {}
        self.enabled:   bool             = enabled
        self.priority:  int              = priority
        self.tools:     List[str]        = tools or []
        self.score_bias: float           = score_bias

        # Embeddings are computed externally and stored here
        self.embeddings: List[List[float]] = []

        # Lifecycle timestamps
        self.created_at: float = time.time()
        self.updated_at: float = time.time()

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #

    def execute(
        self,
        request: HandlerRequest,
        similarity: float = 0.0,
    ) -> HandlerResponse:
        """
        Run the handler, enforce enabled-check, time the call.

        Raises
        ------
        RuntimeError
            If the route is disabled.
        """
        if not self.enabled:
            raise RuntimeError(f"Route '{self.name}' is disabled.")

        if not self.handler.validate(request):
            return HandlerResponse.fail(
                f"Handler validation failed for route '{self.name}'."
            )

        start = time.perf_counter()
        try:
            response = self.handler.handle(request)
        except Exception as exc:
            response = HandlerResponse.fail(
                f"Handler raised {type(exc).__name__}: {exc}"
            )
        finally:
            elapsed = time.perf_counter() - start
            # Attach timing (non-destructive if handler already set it)
            if response.processing_time == 0.0:
                response.processing_time = elapsed

        return response

    async def execute_async(
        self,
        request: HandlerRequest,
        similarity: float = 0.0,
    ) -> HandlerResponse:
        """Async execution path."""
        if not self.enabled:
            raise RuntimeError(f"Route '{self.name}' is disabled.")

        if not self.handler.validate(request):
            return HandlerResponse.fail(
                f"Handler validation failed for route '{self.name}'."
            )

        start = time.perf_counter()
        try:
            response = await self.handler.handle_async(request)
        except Exception as exc:
            response = HandlerResponse.fail(
                f"Handler raised {type(exc).__name__}: {exc}"
            )
        finally:
            elapsed = time.perf_counter() - start
            if response.processing_time == 0.0:
                response.processing_time = elapsed

        return response

    # ------------------------------------------------------------------ #
    # Example management
    # ------------------------------------------------------------------ #

    def add_example(self, example: str) -> bool:
        example = example.strip()
        if example and example not in self.examples:
            self.examples.append(example)
            self.embeddings = []  # Invalidate cached embeddings
            self.updated_at = time.time()
            return True
        return False

    def remove_example(self, example: str) -> bool:
        if example in self.examples:
            self.examples.remove(example)
            self.embeddings = []
            self.updated_at = time.time()
            return True
        return False

    def invalidate_embeddings(self):
        self.embeddings = []
        self.updated_at = time.time()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def enable(self):
        self.enabled    = True
        self.updated_at = time.time()

    def disable(self):
        self.enabled    = False
        self.updated_at = time.time()

    def apply_score_bias(self, delta: float):
        """Called by the FeedbackEngine to nudge the route's score offset."""
        self.score_bias = max(-0.3, min(0.3, self.score_bias + delta))
        self.updated_at = time.time()

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name":        self.name,
            "description": self.description,
            "examples":    self.examples,
            "keywords": {
                "required": self.keywords.required,
                "any_of":   self.keywords.any_of,
                "excluded": self.keywords.excluded,
                "boost":    self.keywords.boost,
            },
            "group":       self.group,
            "tags":        list(self.tags),
            "metadata":    self.metadata,
            "enabled":     self.enabled,
            "priority":    self.priority,
            "tools":       self.tools,
            "score_bias":  self.score_bias,
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        handler: Union[BaseHandler, Callable],
    ) -> "Route":
        kw_data  = data.get("keywords", {})
        keywords = RouteKeywords(
            required = kw_data.get("required", []),
            any_of   = kw_data.get("any_of", []),
            excluded = kw_data.get("excluded", []),
            boost    = kw_data.get("boost", 0.05),
        )
        return cls(
            name        = data["name"],
            description = data["description"],
            handler     = handler,
            examples    = data.get("examples", []),
            keywords    = keywords,
            group       = data.get("group"),
            tags        = set(data.get("tags", [])),
            metadata    = data.get("metadata", {}),
            enabled     = data.get("enabled", True),
            priority    = data.get("priority", 0),
            tools       = data.get("tools", []),
            score_bias  = data.get("score_bias", 0.0),
        )

    # ------------------------------------------------------------------ #
    # Magic
    # ------------------------------------------------------------------ #

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Route) and self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:
        status = "enabled" if self.enabled else "disabled"
        return (
            f"Route(name={self.name!r}, group={self.group!r}, "
            f"examples={len(self.examples)}, priority={self.priority}, "
            f"status={status}, bias={self.score_bias:+.3f})"
        )
