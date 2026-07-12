"""Centralized logging configuration.

Every module obtains its logger via ``logging.getLogger(__name__)``;
this module wires handlers and formatting exactly once.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(
    level: str = "INFO",
    log_format: str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    log_file: Path | None = None,
) -> None:
    """Configure root logging with console and optional file output.

    Args:
        level: Logging level name (e.g. ``"INFO"``).
        log_format: Format string applied to all handlers.
        log_file: Optional file path for persistent logs.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format,
        handlers=handlers,
        force=True,
    )
