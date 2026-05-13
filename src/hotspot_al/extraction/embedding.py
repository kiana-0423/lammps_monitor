"""Embedding hooks for future QM/MM or electrostatic boundary models."""

from __future__ import annotations

from typing import Any

from hotspot_al.models import ExtractedRegion


def build_embedding(region: ExtractedRegion, config: dict[str, Any]) -> dict[str, Any]:
    """Return embedding metadata for a region.

    The baseline project exposes a stable hook but does not implement a
    specific embedding scheme yet.
    """

    return {
        "enabled": bool(config.get("enabled", False)),
        "status": "not_implemented",
    }
