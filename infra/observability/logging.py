"""
SearchBarbara logging infrastructure.

Provides:
- JSONFormatter / ConsoleFormatter — dual-format output
- setup_global_logger()  — process-level logger → logs/searchbarbara.log (JSON) + console (text)
- SessionLogger          — per-session logger → logs/sessions/{session_id}.log (JSON Lines)
- setup_agent_logger()   — backward-compatible shim
"""

import json
import logging
import os
import time
from contextvars import ContextVar
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = PROJECT_ROOT / "logs"
SESSIONS_LOG_DIR = LOGS_DIR / "sessions"
GLOBAL_LOG_FILE = LOGS_DIR / "searchbarbara.log"
AGENT_LOG_FILE = LOGS_DIR / "agent_debug.log"          # kept for compat


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

class JSONFormatter(logging.Formatter):
    """Emit one JSON object per log line — for machine consumption."""

    def format(self, record: logging.LogRecord) -> str:
        obj: Dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Merge structured extras attached via `extra={"structured": {...}}`
        structured = getattr(record, "structured", None)
        if isinstance(structured, dict):
            obj.update(structured)
        return json.dumps(obj, ensure_ascii=False, default=str)


class ConsoleFormatter(logging.Formatter):
    """Human-readable text for the console.

    Appends key structured fields (session_id, stage, node_id, duration_ms,
    tokens, error) when present, so operators can follow session flow
    without digging into JSON files.
    """

    _BASE_FMT = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"
    _DATE_FMT = "%Y-%m-%d %H:%M:%S"

    def __init__(self) -> None:
        super().__init__(fmt=self._BASE_FMT, datefmt=self._DATE_FMT)

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        structured = getattr(record, "structured", None)
        if not isinstance(structured, dict):
            return base
        parts: list[str] = []
        for key in ("session_id", "stage", "node_id", "trace_id"):
            val = structured.get(key)
            if val:
                parts.append(f"{key}={val}")
        dur = structured.get("duration_ms")
        if dur is not None:
            parts.append(f"duration={dur:.0f}ms")
        tokens = structured.get("tokens")
        if isinstance(tokens, dict):
            total = tokens.get("total_tokens")
            if total:
                parts.append(f"tokens={total}")
        err = structured.get("error")
        if err:
            parts.append(f"error={err}")
        if parts:
            return f"{base}  | {' '.join(parts)}"
        return base


# ---------------------------------------------------------------------------
# Global logger
# ---------------------------------------------------------------------------

_global_logger: Optional[logging.Logger] = None


def setup_global_logger() -> logging.Logger:
    """Initialise the process-wide ``searchbarbara`` logger (idempotent).

    • File handler  → ``logs/searchbarbara.log``  (JSON, rotating 50 MB × 5)
    • Console handler → text format
    """
    global _global_logger
    if _global_logger is not None:
        return _global_logger

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    SESSIONS_LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("searchbarbara")
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers on re-import
    if not logger.handlers:
        # --- rotating JSON file handler ---
        fh = RotatingFileHandler(
            str(GLOBAL_LOG_FILE),
            maxBytes=50 * 1024 * 1024,  # 50 MB
            backupCount=5,
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(JSONFormatter())
        logger.addHandler(fh)

        # --- console text handler ---
        ch = logging.StreamHandler()
        level_name = os.getenv("LOG_LEVEL", os.getenv("AGENT_LOG_LEVEL", "INFO")).upper()
        ch.setLevel(getattr(logging, level_name, logging.INFO))
        ch.setFormatter(ConsoleFormatter())
        logger.addHandler(ch)

    _global_logger = logger
    return logger


# ---------------------------------------------------------------------------
# Backward-compatible shim
# ---------------------------------------------------------------------------

def setup_agent_logger() -> logging.Logger:
    """Legacy entry-point — returns the global logger."""
    return setup_global_logger()


# ---------------------------------------------------------------------------
# Session logger
# ---------------------------------------------------------------------------

_current_session_logger: ContextVar["SessionLogger | None"] = ContextVar(
    "_current_session_logger", default=None
)


def get_current_session_logger() -> "Optional[SessionLogger]":
    return _current_session_logger.get()


class SessionLogger:
    """Per-session structured logger.

    Creates ``logs/sessions/{session_id}.log`` (JSON Lines).
    Every record automatically carries *session_id*, *stage*, *node_id*,
    and *trace_id* as structured context.
    """

    def __init__(self, session_id: str, *, trace_id: Optional[str] = None) -> None:
        self.session_id = session_id
        self._stage: str = ""
        self._node_id: str = ""
        self._trace_id: str = trace_id or ""

        # Dedicated logger — propagate=True lets records bubble to the
        # global ``searchbarbara`` console handler.
        self._logger = logging.getLogger(f"searchbarbara.session.{session_id}")
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = True

        # Session file handler — JSON Lines
        SESSIONS_LOG_DIR.mkdir(parents=True, exist_ok=True)
        self._log_path = SESSIONS_LOG_DIR / f"{session_id}.log"
        self._file_handler = logging.FileHandler(
            str(self._log_path), encoding="utf-8"
        )
        self._file_handler.setLevel(logging.DEBUG)
        self._file_handler.setFormatter(JSONFormatter())
        self._logger.addHandler(self._file_handler)

        # Put ourselves in ContextVar so middleware can find us
        self._token = _current_session_logger.set(self)

    # -- context setters --------------------------------------------------

    def set_stage(self, stage: str) -> None:
        self._stage = stage

    def set_node(self, node_id: str) -> None:
        self._node_id = node_id

    def set_trace(self, trace_id: str) -> None:
        self._trace_id = trace_id

    # -- core logging -----------------------------------------------------

    def log(
        self,
        message: str,
        *,
        level: int = logging.INFO,
        stage: Optional[str] = None,
        node_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        input_data: Any = None,
        output_data: Any = None,
        duration_ms: Optional[float] = None,
        tokens: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        **extra_fields: Any,
    ) -> None:
        """Emit one structured log entry."""
        structured: Dict[str, Any] = {
            "session_id": self.session_id,
            "stage": stage or self._stage,
            "node_id": node_id or self._node_id,
            "trace_id": trace_id or self._trace_id,
        }
        if input_data is not None:
            structured["input"] = input_data
        if output_data is not None:
            structured["output"] = output_data
        if duration_ms is not None:
            structured["duration_ms"] = round(duration_ms, 2)
        if tokens is not None:
            structured["tokens"] = tokens
        if error is not None:
            structured["error"] = error
        structured.update(extra_fields)

        self._logger.log(level, message, extra={"structured": structured})

    # Convenience shortcuts
    def info(self, message: str, **kwargs: Any) -> None:
        self.log(message, level=logging.INFO, **kwargs)

    def debug(self, message: str, **kwargs: Any) -> None:
        self.log(message, level=logging.DEBUG, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        self.log(message, level=logging.WARNING, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        self.log(message, level=logging.ERROR, **kwargs)

    # -- lifecycle --------------------------------------------------------

    def close(self) -> None:
        """Flush and detach the file handler."""
        try:
            self._file_handler.flush()
            self._file_handler.close()
            self._logger.removeHandler(self._file_handler)
        except Exception:
            pass
        # Reset ContextVar
        try:
            _current_session_logger.set(None)
        except Exception:
            pass

    @property
    def log_path(self) -> Path:
        return self._log_path
