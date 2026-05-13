"""Configuration loading and shallow-merge helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "default.yaml"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML document into a dictionary."""

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries."""

    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: str | Path | None = None, *, base_path: str | Path | None = None) -> dict[str, Any]:
    """Load the default config and optionally merge an override file."""

    base = load_yaml(base_path or DEFAULT_CONFIG_PATH)
    if path is None:
        return base
    return merge_dicts(base, load_yaml(path))
