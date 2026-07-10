"""Shared protocol definitions for the hotspot AL pipeline."""

from __future__ import annotations

from typing import Protocol

from hotspot_al.backends.base import DFTBackend, MDBackend, MLIPBackend, SchedulerBackend
from hotspot_al.models import FrameData


class FrameSource(Protocol):
    """Any object that yields ``FrameData`` frames."""

    def next_frame(self, timeout: float | None = None) -> FrameData | None:
        """Return the next available frame or ``None`` when exhausted."""


# Backward-compatible name; new code should use MLIPBackend explicitly.
ForceBackend = MLIPBackend

__all__ = ["DFTBackend", "ForceBackend", "FrameSource", "MDBackend", "MLIPBackend", "SchedulerBackend"]
