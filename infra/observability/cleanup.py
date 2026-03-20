"""
Session log cleanup utility.

Deletes session log files older than a configurable threshold
(default: 30 days).  Called at startup from ``run_web.py``.
"""

import logging
import os
import time
from pathlib import Path

log = logging.getLogger("searchbarbara.cleanup")

# Default: 30 days
SESSION_LOG_MAX_AGE_DAYS = int(os.getenv("SESSION_LOG_MAX_AGE_DAYS", "30"))

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SESSIONS_LOG_DIR = _PROJECT_ROOT / "logs" / "sessions"


def cleanup_old_session_logs(
    max_age_days: int | None = None,
    log_dir: Path | None = None,
) -> int:
    """Remove session log files older than *max_age_days*.

    Returns the number of files deleted.
    """
    age_days = max_age_days if max_age_days is not None else SESSION_LOG_MAX_AGE_DAYS
    target_dir = log_dir or _SESSIONS_LOG_DIR
    if not target_dir.is_dir():
        return 0

    cutoff = time.time() - age_days * 86400
    deleted = 0
    for f in target_dir.iterdir():
        if not f.is_file():
            continue
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
                deleted += 1
        except Exception as exc:
            log.warning("Failed to delete old session log %s: %s", f.name, exc)

    if deleted:
        log.info(
            "Cleaned up %d session log(s) older than %d days from %s",
            deleted,
            age_days,
            target_dir,
        )
    return deleted
