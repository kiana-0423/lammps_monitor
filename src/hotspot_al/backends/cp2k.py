"""CP2K backend skeleton."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

from hotspot_al.cp2k.cp2k_force_parser import parse_cp2k_forces


class CP2KBackend:
    """Thin CP2K adapter placeholder for future runtime orchestration."""

    def __init__(self, *, executable: str | Path | None = None) -> None:
        self.executable = None if executable is None else str(executable)

    def evaluate_forces(
        self,
        atoms: Atoms,
        *,
        config: dict[str, Any] | None = None,
        model_path: str | Path | None = None,
    ) -> np.ndarray:
        """CP2K force evaluation needs a job runner and is not wired by default."""

        msg = "CP2KBackend.evaluate_forces requires an external CP2K job runner; use integration helpers or provide orchestration."
        raise NotImplementedError(msg)

    def parse_forces(self, output_path: str | Path) -> np.ndarray:
        """Parse CP2K forces from an existing output file."""

        return parse_cp2k_forces(output_path)

