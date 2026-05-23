"""Shared backend contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np
from ase import Atoms


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

