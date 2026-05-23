"""LAMMPS backend skeleton."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

from hotspot_al.io.lammps_reader import read_dump


class LAMMPSBackend:
    """Thin LAMMPS adapter placeholder for future runtime orchestration."""

    def __init__(self, *, executable: str | Path | None = None) -> None:
        self.executable = None if executable is None else str(executable)

    def evaluate_forces(
        self,
        atoms: Atoms,
        *,
        config: dict[str, Any] | None = None,
        model_path: str | Path | None = None,
    ) -> np.ndarray:
        """LAMMPS force evaluation needs an input generator and is not wired by default."""

        msg = "LAMMPSBackend.evaluate_forces requires an external LAMMPS input/runtime adapter."
        raise NotImplementedError(msg)

    def read_dump_forces(
        self,
        dump_path: str | Path,
        *,
        type_map: dict[int, str] | None = None,
    ) -> np.ndarray:
        """Read forces from the first frame of a LAMMPS dump."""

        frames = read_dump(dump_path, type_map=type_map)
        if not frames or frames[0].forces is None:
            msg = f"No force-bearing LAMMPS frame found in {dump_path}."
            raise ValueError(msg)
        return frames[0].forces

