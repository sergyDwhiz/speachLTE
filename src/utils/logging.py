"""
Structured logging configuration using python logging.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def configure_logging(level: int = logging.INFO, log_dir: Optional[Path] = None) -> None:
    """Configure root logger with console and optional file handler."""
    handlers = [logging.StreamHandler()]
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_dir / "speachlte.log"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=handlers,
    )
