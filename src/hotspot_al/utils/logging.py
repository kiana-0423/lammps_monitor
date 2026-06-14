"""Project logging helpers."""

from __future__ import annotations

import logging
import os
import json
from pathlib import Path
from typing import Any


class JsonFormatter(logging.Formatter):
    """Format log records as one compact JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "time": self.formatTime(record, self.datefmt),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def get_logger(
    name: str = "hotspot_al",
    level: int | str | None = None,
    *,
    log_file: str | Path | None = None,
    log_format: str | None = None,
) -> logging.Logger:
    """Return a configured project logger."""

    logger = logging.getLogger(name)
    resolved_level = _resolve_level(level)
    formatter = _make_formatter(log_format)
    if not any(getattr(handler, "_hotspot_console", False) for handler in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        handler.setLevel(resolved_level)
        handler._hotspot_console = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    if log_file is not None and not _has_file_handler(logger, Path(log_file)):
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
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
    log_format = os.environ.get("HOTSPOT_AL_LOG_FORMAT") or logging_cfg.get("format")
    return get_logger(name, level=level, log_file=log_file, log_format=log_format)


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


def _make_formatter(log_format: str | None) -> logging.Formatter:
    if str(log_format or "text").lower() == "json":
        return JsonFormatter()
    return logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
