"""
Lightweight in-process metrics collection for SearchBarbara.

Provides:
- MetricsCollector — per-session counters and timing histograms for summary reporting
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional


class MetricsCollector:
    """Lightweight per-session metrics: counters, gauges, and timing histograms.

    Thread-safe — safe to update from the agent worker thread and read from
    the main thread / API layer.

    Usage::

        mc = MetricsCollector()
        mc.increment("llm_calls")
        mc.increment("tokens.prompt", 342)
        mc.record_timing("llm.latency_ms", 1823.5)
        summary = mc.summary()
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Dict[str, float] = defaultdict(float)
        self._timings: Dict[str, List[float]] = defaultdict(list)

    # -- counters ---------------------------------------------------------

    def increment(self, name: str, amount: float = 1) -> None:
        with self._lock:
            self._counters[name] += amount

    def get_counter(self, name: str) -> float:
        with self._lock:
            return self._counters.get(name, 0)

    # -- timing histograms ------------------------------------------------

    def record_timing(self, name: str, value_ms: float) -> None:
        with self._lock:
            self._timings[name].append(value_ms)

    # -- summary ----------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return a snapshot of all metrics for session-level reporting."""
        with self._lock:
            result: Dict[str, Any] = {
                "counters": dict(self._counters),
            }
            timing_summaries: Dict[str, Dict[str, float]] = {}
            for name, values in self._timings.items():
                if not values:
                    continue
                sorted_v = sorted(values)
                n = len(sorted_v)
                timing_summaries[name] = {
                    "count": n,
                    "min": sorted_v[0],
                    "max": sorted_v[-1],
                    "mean": sum(sorted_v) / n,
                    "p50": sorted_v[n // 2],
                    "p95": sorted_v[int(n * 0.95)] if n >= 2 else sorted_v[-1],
                    "p99": sorted_v[int(n * 0.99)] if n >= 2 else sorted_v[-1],
                }
            result["timings"] = timing_summaries
            return result

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._timings.clear()
