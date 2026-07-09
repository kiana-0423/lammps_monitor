"""Shared numeric codes for extracted-region labels."""

from __future__ import annotations

import json
from collections.abc import Iterable

import numpy as np

REGION_CODE_MAP: dict[str, int] = {
    "core": 0,
    "label_core": 0,
    "inner_buffer": 1,
    "outer_buffer": 2,
    "boundary": 3,
    "frozen_boundary": 3,
    "h_cap": 4,
}

REGION_LABEL_MAP: dict[int, str] = {
    0: "core",
    1: "inner_buffer",
    2: "outer_buffer",
    3: "boundary",
    4: "h_cap",
}


def region_codes_for_labels(labels: Iterable[str]) -> np.ndarray:
    """Convert region labels into stable integer codes."""

    return np.asarray([REGION_CODE_MAP.get(label, -1) for label in labels], dtype=int)


def region_label_map_json() -> str:
    """Return the extxyz metadata representation of region-code labels."""

    return json.dumps(REGION_LABEL_MAP)
