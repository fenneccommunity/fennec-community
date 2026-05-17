"""
===============
Lightweight execution tracer that builds a structured execution tree.

Every chain step emits a TraceSpan which is collected into an
ExecutionTrace.  The trace can be inspected, pretty-printed, or
serialised to JSON for external observability systems.

Example
-------
>>> tracer = ExecutionTracer()
>>> with tracer.span("MyChain") as span:
...     result = do_work()
...     span.set_output(result)
>>> print(tracer.root.pretty())
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional


@dataclass
class TraceSpan:
    """
    A single node in the execution tree.
    """

    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    chain_name: str = ""
    start_time: float = field(default_factory=time.perf_counter)
    end_time: Optional[float] = None
    success: bool = True
    error: Optional[str] = None
    input_summary: str = ""
    output_summary: str = ""
    children: List["TraceSpan"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def finish(self, success: bool = True, error: Optional[str] = None) -> None:
        """Mark the span as complete."""
        self.end_time = time.perf_counter()
        self.success = success
        self.error = error

    @property
    def duration_ms(self) -> float:
        """Elapsed wall-clock time in milliseconds."""
        end = self.end_time if self.end_time is not None else time.perf_counter()
        return (end - self.start_time) * 1000

    def set_input(self, data: Any) -> None:
        self.input_summary = str(data)[:120]

    def set_output(self, data: Any) -> None:
        self.output_summary = str(data)[:120]

    def add_child(self, child: "TraceSpan") -> None:
        self.children.append(child)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "span_id": self.span_id,
            "chain": self.chain_name,
            "duration_ms": round(self.duration_ms, 2),
            "success": self.success,
            "error": self.error,
            "input": self.input_summary,
            "output": self.output_summary,
            "metadata": self.metadata,
            "steps": [c.to_dict() for c in self.children],
        }

    def pretty(self, indent: int = 0) -> str:
        """Human-readable tree representation."""
        status = "✅" if self.success else "❌"
        lines = [
            " " * indent
            + f"{status} [{self.chain_name}] {self.duration_ms:.1f}ms"
            + (f"  error={self.error}" if self.error else "")
        ]
        for child in self.children:
            lines.append(child.pretty(indent + 4))
        return "\n".join(lines)


class ExecutionTracer:
    """
    Manages the current execution tree during a chain run.

    Usage
    -----
    >>> tracer = ExecutionTracer()
    >>> with tracer.span("RootChain") as span:
    ...     span.set_input("my data")
    ...     # child spans are created automatically by nested chains
    """

    def __init__(self) -> None:
        self.root: Optional[TraceSpan] = None
        self._stack: List[TraceSpan] = []

    @contextmanager
    def span(self, chain_name: str) -> Generator[TraceSpan, None, None]:
        """
        Context manager that opens a new span and links it into the tree.
        """
        new_span = TraceSpan(chain_name=chain_name)

        if self._stack:
            # Attach as child of the current active span
            self._stack[-1].add_child(new_span)
        else:
            self.root = new_span

        self._stack.append(new_span)
        try:
            yield new_span
            new_span.finish(success=True)
        except Exception as exc:
            new_span.finish(success=False, error=str(exc))
            raise
        finally:
            self._stack.pop()

    def current_span(self) -> Optional[TraceSpan]:
        return self._stack[-1] if self._stack else None

    def to_dict(self) -> Dict[str, Any]:
        if self.root is None:
            return {}
        return self.root.to_dict()

    def pretty(self) -> str:
        if self.root is None:
            return "(no trace)"
        return self.root.pretty()
