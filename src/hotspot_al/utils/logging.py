"""Project logging helpers."""

from __future__ import annotations

import logging


def get_logger(name: str = "hotspot_al", level: int = logging.INFO) -> logging.Logger:
    """Return a configured project logger."""

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
