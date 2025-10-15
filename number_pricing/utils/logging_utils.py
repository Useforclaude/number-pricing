"""Centralised logging utilities for the number pricing project."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Optional

from number_pricing.config import CONFIG

_LOGGING_INITIALISED = False


def _create_file_handler(log_path: Path) -> logging.Handler:
    """Create a rotating handler using the scheme defined in the config."""
    rotation = CONFIG.logging.rotation.lower()
    if rotation in {"day", "midnight", "daily"}:
        handler = TimedRotatingFileHandler(
            filename=log_path,
            when="midnight",
            backupCount=CONFIG.logging.backup_count,
            encoding="utf-8",
        )
    else:
        handler = RotatingFileHandler(
            filename=log_path,
            maxBytes=CONFIG.logging.max_bytes,
            backupCount=CONFIG.logging.backup_count,
            encoding="utf-8",
        )
    return handler


def setup_logging(force: bool = False) -> None:
    """Configure the root logger according to CONFIG.logging."""
    global _LOGGING_INITIALISED
    if _LOGGING_INITIALISED and not force:
        return

    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(CONFIG.logging.level)

    formatter = logging.Formatter(CONFIG.logging.format)

    log_file = CONFIG.paths.logs_dir / "number_pricing.log"
    file_handler = _create_file_handler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.propagate = CONFIG.logging.propagate
    _LOGGING_INITIALISED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger instance, ensuring the logging system is initialised."""
    setup_logging()
    return logging.getLogger(name)

