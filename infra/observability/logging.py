import logging
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = PROJECT_ROOT / "logs"
AGENT_LOG_FILE = LOGS_DIR / "agent_debug.log"


def setup_agent_logger() -> logging.Logger:
    logger = logging.getLogger("searchbarbara.agent")
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(str(AGENT_LOG_FILE), encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    level_name = os.getenv("AGENT_LOG_LEVEL", "WARNING").upper()
    console_handler.setLevel(getattr(logging, level_name, logging.WARNING))
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)
    return logger
