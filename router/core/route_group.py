"""

RouteGroup provides the "module" abstraction.
Each group is a self-contained namespace of routes that can be:
  - registered/deregistered as a unit
  - given an intent description for the top-level intent router
  - restricted by tags or priority
  - used as a sub-router in its own right

Built-in groups ship for the most common LLM application patterns
(RAG, Tool use, Chat) and serve as templates for custom groups.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Union

from .base import BaseHandler, HandlerRequest
from .route import Route, RouteKeywords


# ---------------------------------------------------------------------------
# RouteGroup
# ---------------------------------------------------------------------------

class RouteGroup:
    """
    A named collection of :class:`Route` objects sharing a common
    *intent* (e.g. "answer questions from documents", "use external tools").

    The group itself behaves like a lightweight sub-router — the parent
    ``HierarchicalRouter`` first identifies the best group, then delegates
    within it.

    Parameters
    ----------
    name            : Unique name (e.g. "rag", "tools", "chat").
    description     : Intent description used for top-level routing.
    intent_examples : Sample queries that signal this group's intent.
    intent_keywords : Keyword signals for the group-level selector.
    priority        : Group-level tie-breaker (higher wins).
    enabled         : Whether this group participates in routing.
    metadata        : Arbitrary labels / config.
    """

    def __init__(
        self,
        name:             str,
        description:      str,
        intent_examples:  Optional[List[str]]        = None,
        intent_keywords:  Optional[RouteKeywords]    = None,
        priority:         int                        = 0,
        enabled:          bool                       = True,
        metadata:         Optional[Dict[str, Any]]   = None,
    ):
        if not name or not name.strip():
            raise ValueError("RouteGroup name cannot be empty.")

        self.name             = name.strip()
        self.description      = description.strip()
        self.intent_examples: List[str]      = list(dict.fromkeys(intent_examples or []))
        self.intent_keywords: RouteKeywords  = intent_keywords or RouteKeywords()
        self.priority:        int            = priority
        self.enabled:         bool           = enabled
        self.metadata:        Dict[str, Any] = metadata or {}

        self._routes: Dict[str, Route] = {}

        # Group-level embedding (computed by the router from intent_examples)
        self.embeddings: List[List[float]] = []

        self.created_at = time.time()
        self.updated_at = time.time()

    # ------------------------------------------------------------------ #
    # Route management
    # ------------------------------------------------------------------ #

    def add_route(self, route: Route) -> "RouteGroup":
        """Register a route.  Raises if name already exists (use update)."""
        if route.name in self._routes:
            raise ValueError(
                f"Route '{route.name}' already exists in group '{self.name}'. "
                f"Use update_route() to replace it."
            )
        route.group = self.name
        self._routes[route.name] = route
        self.updated_at = time.time()
        return self          # fluent API

    def update_route(self, route: Route) -> "RouteGroup":
        """Add or replace a route."""
        route.group = self.name
        self._routes[route.name] = route
        self.updated_at = time.time()
        return self

    def remove_route(self, name: str) -> bool:
        if name in self._routes:
            del self._routes[name]
            self.updated_at = time.time()
            return True
        return False

    def get_route(self, name: str) -> Optional[Route]:
        return self._routes.get(name)

    def get_routes(self, enabled_only: bool = True) -> List[Route]:
        routes = list(self._routes.values())
        if enabled_only:
            routes = [r for r in routes if r.enabled]
        return sorted(routes, key=lambda r: r.priority, reverse=True)

    def get_by_tag(self, tag: str) -> List[Route]:
        return [r for r in self._routes.values() if tag in r.tags]

    # ------------------------------------------------------------------ #
    # Convenience builder methods (fluent factory)
    # ------------------------------------------------------------------ #

    def route(
        self,
        name:        str,
        description: str,
        examples:    Optional[List[str]]     = None,
        keywords:    Optional[RouteKeywords] = None,
        priority:    int                     = 0,
        tags:        Optional[Set[str]]      = None,
        tools:       Optional[List[str]]     = None,
    ):
        """Decorator that registers the decorated class/function as a handler."""
        def decorator(handler_or_cls):
            if isinstance(handler_or_cls, type) and issubclass(handler_or_cls, BaseHandler):
                handler = handler_or_cls()
            elif isinstance(handler_or_cls, BaseHandler):
                handler = handler_or_cls
            elif callable(handler_or_cls):
                from .base import CallableHandler
                handler = CallableHandler(handler_or_cls)
            else:
                raise TypeError(f"Cannot use {type(handler_or_cls)} as a handler.")

            self.add_route(Route(
                name        = name,
                description = description,
                handler     = handler,
                examples    = examples,
                keywords    = keywords,
                priority    = priority,
                tags        = tags,
                tools       = tools,
            ))
            return handler_or_cls
        return decorator

    # ------------------------------------------------------------------ #
    # Group lifecycle
    # ------------------------------------------------------------------ #

    def enable(self):
        self.enabled    = True
        self.updated_at = time.time()

    def disable(self):
        self.enabled    = False
        self.updated_at = time.time()

    def invalidate_embeddings(self):
        """Force re-encoding of all routes in this group."""
        self.embeddings = []
        for route in self._routes.values():
            route.invalidate_embeddings()
        self.updated_at = time.time()

    # ------------------------------------------------------------------ #
    # Info
    # ------------------------------------------------------------------ #

    def summary(self) -> Dict[str, Any]:
        return {
            "name":         self.name,
            "description":  self.description,
            "enabled":      self.enabled,
            "priority":     self.priority,
            "route_count":  len(self._routes),
            "routes":       [r.name for r in self._routes.values()],
            "metadata":     self.metadata,
        }

    def __len__(self) -> int:
        return len(self._routes)

    def __iter__(self):
        return iter(self._routes.values())

    def __contains__(self, name: str) -> bool:
        return name in self._routes

    def __repr__(self) -> str:
        status = "enabled" if self.enabled else "disabled"
        return (
            f"RouteGroup(name={self.name!r}, routes={len(self._routes)}, "
            f"status={status}, priority={self.priority})"
        )


# ---------------------------------------------------------------------------
# Built-in Group Templates
# ---------------------------------------------------------------------------

def make_rag_group(priority: int = 10) -> RouteGroup:
    """
    Template: Retrieval-Augmented Generation group.
    Sub-routes should retrieve documents and synthesise answers.
    """
    return RouteGroup(
        name        = "rag",
        description = "Answer questions by retrieving and synthesising from documents, knowledge bases, or vector stores.",
        intent_examples = [
            "What does the policy say about refunds?",
            "Find information about our return procedure",
            "According to the manual, how do I reset the device?",
            "Search the knowledge base for onboarding steps",
            "What are the terms and conditions?",
            "Retrieve the latest quarterly report",
        ],
        intent_keywords = RouteKeywords(
            any_of   = ["according", "find", "search", "retrieve", "document", "manual", "policy", "report"],
            excluded = ["write", "create", "generate", "code"],
            boost    = 0.08,
        ),
        priority = priority,
        metadata = {"type": "rag"},
    )


def make_tools_group(priority: int = 20) -> RouteGroup:
    """
    Template: Tool/function-calling group.
    Sub-routes call external APIs, run code, query databases, etc.
    """
    return RouteGroup(
        name        = "tools",
        description = "Execute actions using tools: call APIs, run code, query databases, send emails, manage calendars.",
        intent_examples = [
            "Send an email to John about the meeting",
            "What is the weather in Cairo right now?",
            "Execute this Python script",
            "Add a task to my calendar for tomorrow",
            "Search the web for the latest news",
            "Query the database for user records",
            "Run the data pipeline",
        ],
        intent_keywords = RouteKeywords(
            any_of   = ["send", "run", "execute", "call", "query", "schedule", "create", "delete", "update", "weather", "search web"],
            boost    = 0.10,
        ),
        priority = priority,
        metadata = {"type": "tools"},
    )


def make_chat_group(priority: int = 0) -> RouteGroup:
    """
    Template: Conversational chat group.
    Sub-routes handle general conversation, QA, writing, coding, etc.
    This is typically the lowest-priority catch-all group.
    """
    return RouteGroup(
        name        = "chat",
        description = "Handle general conversation, creative writing, coding help, and open-ended questions.",
        intent_examples = [
            "How are you?",
            "Tell me a joke",
            "Help me write a cover letter",
            "Explain quantum computing simply",
            "What is the capital of France?",
            "Write a Python function to sort a list",
        ],
        intent_keywords = RouteKeywords(
            boost = 0.02,
        ),
        priority = priority,
        metadata = {"type": "chat"},
    )
