"""

Defines the fundamental contracts (abstract base classes & value objects)
that every component in the system must satisfy.

Design goals:
  - Handlers are explicit objects, not bare callables → easier to test/mock
  - Request/Response are typed dataclasses → IDE-friendly, schema-validated
  - Async is a first-class citizen alongside sync
"""
from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Iterator, Optional


# ---------------------------------------------------------------------------
# Value Objects: Request / Response
# ---------------------------------------------------------------------------

@dataclass
class HandlerRequest:
    """
    Unified input object passed to every handler.

    Fields
    ------
    query       : The raw natural-language query from the user.
    context     : Arbitrary key-value context (conversation history, user profile…).
    request_id  : UUID for correlation across logs and traces.
    timestamp   : Unix epoch of when the request was created.
    metadata    : Router-internal metadata (route name, confidence score, etc.).
                  Handlers may read but should not depend on this in business logic.
    """
    query:      str
    context:    Dict[str, Any]  = field(default_factory=dict)
    request_id: str             = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp:  float           = field(default_factory=time.time)
    metadata:   Dict[str, Any]  = field(default_factory=dict)

    def with_metadata(self, **kwargs) -> "HandlerRequest":
        """Return a copy with extra metadata merged in (immutable-style update)."""
        new_meta = {**self.metadata, **kwargs}
        return HandlerRequest(
            query=self.query,
            context=self.context,
            request_id=self.request_id,
            timestamp=self.timestamp,
            metadata=new_meta,
        )


@dataclass
class HandlerResponse:
    """
    Unified output object returned by every handler.

    Fields
    ------
    content         : The payload (string, dict, list, model output, …).
    success         : False if the handler experienced a handled error.
    error           : Human-readable error message (only set when success=False).
    metadata        : Handler-supplied annotations (sources, token count, …).
    processing_time : Seconds the handler took (set by ExecutionEngine).
    """
    content:          Any
    success:          bool              = True
    error:            Optional[str]     = None
    metadata:         Dict[str, Any]    = field(default_factory=dict)
    processing_time:  float             = 0.0

    # ---- Convenience constructors ----------------------------------------

    @classmethod
    def ok(cls, content: Any, **meta) -> "HandlerResponse":
        return cls(content=content, success=True, metadata=meta)

    @classmethod
    def fail(cls, error: str, **meta) -> "HandlerResponse":
        return cls(content=None, success=False, error=error, metadata=meta)

    def __bool__(self) -> bool:
        return self.success


# ---------------------------------------------------------------------------
# Base Handler
# ---------------------------------------------------------------------------

class BaseHandler(ABC):
    """
    Abstract base for all route handlers.

    Every handler must implement ``handle()``.  The async variant
    ``handle_async()`` defaults to running the sync version but can be
    overridden for true async I/O (LLM calls, DB queries, etc.).

    Streaming is supported via ``stream()`` / ``stream_async()`` which
    yield partial results — useful for SSE or chunked HTTP responses.

    Example
    -------
    ::

        class WeatherHandler(BaseHandler):
            def handle(self, request: HandlerRequest) -> HandlerResponse:
                city = extract_city(request.query)
                data = fetch_weather(city)
                return HandlerResponse.ok(data)
    """

    # ---- Required ----------------------------------------------------------

    @abstractmethod
    def handle(self, request: HandlerRequest) -> HandlerResponse:
        """Synchronously process a request."""
        ...

    # ---- Optional overrides ------------------------------------------------

    async def handle_async(self, request: HandlerRequest) -> HandlerResponse:
        """
        Asynchronously process a request.
        Default: delegates to the sync version.
        Override for true async I/O.
        """
        return self.handle(request)

    def validate(self, request: HandlerRequest) -> bool:
        """
        Pre-execution validation hook.
        Return False to abort execution (router will fall to next candidate).
        """
        return True

    def stream(self, request: HandlerRequest) -> Iterator[str]:
        """
        Yield response chunks for streaming.
        Default: yield the entire content as one chunk.
        """
        response = self.handle(request)
        yield str(response.content)

    async def stream_async(self, request: HandlerRequest) -> AsyncIterator[str]:
        """Async streaming variant."""
        response = await self.handle_async(request)
        yield str(response.content)

    def __call__(self, request: HandlerRequest) -> HandlerResponse:
        """Allow handler instances to be called directly."""
        return self.handle(request)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ---------------------------------------------------------------------------
# Callable Adapter  (backward-compatibility shim)
# ---------------------------------------------------------------------------

class CallableHandler(BaseHandler):
    """
    Wraps a plain ``Callable(query: str) -> Any`` so that legacy handlers
    from the original router still work without modification.

    Usage
    -----
    ::

        old_fn = lambda q: {"answer": q}
        handler = CallableHandler(old_fn)
    """

    def __init__(self, fn):
        if not callable(fn):
            raise TypeError(f"Expected a callable, got {type(fn).__name__}")
        self._fn = fn

    def handle(self, request: HandlerRequest) -> HandlerResponse:
        try:
            result = self._fn(request.query)
            return HandlerResponse.ok(result)
        except Exception as exc:
            return HandlerResponse.fail(str(exc))

    def __repr__(self) -> str:
        name = getattr(self._fn, "__name__", repr(self._fn))
        return f"CallableHandler({name})"


# ---------------------------------------------------------------------------
# Base Router Interface
# ---------------------------------------------------------------------------

class BaseRouter(ABC):
    """Abstract contract that both the top-level HierarchicalRouter and
    inner sub-routers must satisfy.  Allows sub-routers to be nested
    arbitrarily deep.
    """

    @abstractmethod
    def route(self, request: HandlerRequest) -> "RoutingResult":  # type: ignore[name-defined]
        ...

    @abstractmethod
    async def route_async(self, request: HandlerRequest) -> "RoutingResult":  # type: ignore[name-defined]
        ...

    @abstractmethod
    def register(self, route: "Route") -> None:  # type: ignore[name-defined]
        ...
