"""Shared protocol definitions for the hotspot AL pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np
from ase import Atoms

from hotspot_al.models import FrameData


class FrameSource(Protocol):
    """Any object that yields ``FrameData`` frames."""

    def next_frame(self, timeout: float | None = None) -> FrameData | None:
        """Return the next available frame or ``None`` when exhausted."""


class ForceBackend(Protocol):
    """Protocol for runtime backends that can evaluate atomic forces."""

    def evaluate_forces(
        self,
        atoms: Atoms,
        *,
        config: dict[str, Any] | None = None,
        model_path: str | Path | None = None,
    ) -> np.ndarray:
        """Return forces with shape ``(n_atoms, 3)``."""
