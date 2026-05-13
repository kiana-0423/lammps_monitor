"""Parse atomic forces from CP2K output files."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from ase.units import Bohr, Hartree


_CP2K_FORCE_HEADER = re.compile(r"ATOMIC FORCES in \[a\.u\.\]", re.IGNORECASE)
_CP2K_FORCE_LINE = re.compile(
    r"^\s*\d+\s+\d+\s+[A-Za-z]{1,3}\s+([\-0-9Ee+.]+)\s+([\-0-9Ee+.]+)\s+([\-0-9Ee+.]+)\s*$"
)


def parse_cp2k_forces(path: str | Path) -> np.ndarray:
    """Parse a CP2K force block and return forces in eV/Å."""

    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    capture = False
    forces: list[list[float]] = []
    for line in lines:
        if _CP2K_FORCE_HEADER.search(line):
            capture = False
            forces = []
            continue
        if "Atom   Kind   Element" in line:
            capture = True
            continue
        if capture:
            if not line.strip() or line.lstrip().startswith("SUM"):
                capture = False
                continue
            match = _CP2K_FORCE_LINE.match(line)
            if match:
                forces.append([float(match.group(1)), float(match.group(2)), float(match.group(3))])
    if not forces:
        msg = f"No CP2K force block found in {path}"
        raise ValueError(msg)
    return np.asarray(forces, dtype=float) * (Hartree / Bohr)
