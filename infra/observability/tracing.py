"""
Tracing helpers for SearchBarbara.

Provides:
- generate_trace_id()  — 16-char hex trace ID
- SpanContext          — context-manager that times a code block and logs via SessionLogger
- StageTracker        — tracks parent→child DFS stage relationships
"""

from __future__ import annotations

import secrets
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from infra.observability.logging import SessionLogger


def generate_trace_id() -> str:
    """Return a random 16-character hex trace ID."""
    return secrets.token_hex(8)


# ---------------------------------------------------------------------------
# SpanContext
# ---------------------------------------------------------------------------

class SpanContext:
    """Context-manager that records timing + structured data for a code block.

    Usage::

        with SpanContext("synthesis", session_logger, node_id="q3") as span:
            result = synthesize(...)
            span.set("tokens", {"prompt_tokens": 120, "completion_tokens": 50})
        # → automatically logs: stage=synthesis, duration_ms=xxx, node_id=q3
    """

    def __init__(
        self,
        name: str,
        session_logger: Optional["SessionLogger"] = None,
        *,
        node_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        **initial_fields: Any,
    ) -> None:
        self.name = name
        self._session_logger = session_logger
        self._node_id = node_id or ""
        self._trace_id = trace_id or ""
        self._fields: Dict[str, Any] = dict(initial_fields)
        self._start: float = 0.0
        self._duration_ms: float = 0.0

    def set(self, key: str, value: Any) -> None:
        """Attach an extra structured field to this span."""
        self._fields[key] = value

    @property
    def duration_ms(self) -> float:
        return self._duration_ms

    def __enter__(self) -> "SpanContext":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._duration_ms = (time.perf_counter() - self._start) * 1000
        if self._session_logger is not None:
            error_str: Optional[str] = None
            if exc_val is not None:
                error_str = f"{exc_type.__name__}: {exc_val}"
            self._session_logger.log(
                f"span:{self.name}",
                stage=self.name,
                node_id=self._node_id,
                trace_id=self._trace_id,
                duration_ms=self._duration_ms,
                error=error_str,
                **self._fields,
            )


# ---------------------------------------------------------------------------
# StageTracker — DFS tree parent-child tracking
# ---------------------------------------------------------------------------

class StageTracker:
    """Maintains a stack of span IDs to track DFS parent-child stage relationships."""

    def __init__(self) -> None:
        self._stack: List[str] = []

    def push(self, span_id: str) -> None:
        self._stack.append(span_id)

    def pop(self) -> Optional[str]:
        return self._stack.pop() if self._stack else None

    @property
    def current(self) -> Optional[str]:
        return self._stack[-1] if self._stack else None

    @property
    def parent(self) -> Optional[str]:
        return self._stack[-2] if len(self._stack) >= 2 else None

    @property
    def depth(self) -> int:
        return len(self._stack)

    def path(self) -> List[str]:
        """Return the full span path from root to current."""
        return list(self._stack)
