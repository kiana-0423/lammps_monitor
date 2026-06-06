"""Project logging helpers."""

from __future__ import annotations

import logging
import os
from pathlib import Path


def get_logger(
    name: str = "hotspot_al",
    level: int | str | None = None,
    *,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Return a configured project logger."""

    logger = logging.getLogger(name)
    resolved_level = _resolve_level(level)
    if not any(getattr(handler, "_hotspot_console", False) for handler in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
        handler.setLevel(resolved_level)
        handler._hotspot_console = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    if log_file is not None and not _has_file_handler(logger, Path(log_file)):
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
    logger.setLevel(resolved_level)
    logger.propagate = False
    return logger


def configure_logging(config: dict | None = None, *, name: str = "hotspot_al") -> logging.Logger:
    """Configure the project logger from config and environment variables."""

    logging_cfg = (config or {}).get("logging", {})
    level = logging_cfg.get("level")
    log_file = os.environ.get("HOTSPOT_AL_LOG_FILE") or logging_cfg.get("file")
    return get_logger(name, level=level, log_file=log_file)


def _resolve_level(level: int | str | None) -> int:
    raw_level = os.environ.get("LOG_LEVEL") if level is None else level
    if raw_level is None:
        return logging.INFO
    if isinstance(raw_level, int):
        return raw_level
    return getattr(logging, str(raw_level).upper(), logging.INFO)


def _has_file_handler(logger: logging.Logger, path: Path) -> bool:
    resolved = path.resolve()
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename).resolve() == resolved:
            return True
    return False
