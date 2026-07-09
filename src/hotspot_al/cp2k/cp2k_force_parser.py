"""Parse atomic forces from CP2K output files."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from ase.units import Bohr, Hartree

from hotspot_al.exceptions import DataError


_CP2K_FORCE_HEADER = re.compile(r"ATOMIC FORCES in \[a\.u\.\]", re.IGNORECASE)
_CP2K_FORCE_LINE = re.compile(
    r"^\s*\d+\s+\d+\s+[A-Za-z]{1,3}\s+([\-0-9Ee+.]+)\s+([\-0-9Ee+.]+)\s+([\-0-9Ee+.]+)\s*$"
)
_CP2K_FORCE_EVAL_HEADER = re.compile(r"FORCES\|\s+Atomic forces \[hartree/bohr\]", re.IGNORECASE)
_CP2K_FORCE_EVAL_LINE = re.compile(
    r"^\s*FORCES\|\s+\d+\s+([\-0-9Ee+.]+)\s+([\-0-9Ee+.]+)\s+([\-0-9Ee+.]+)\s+[\-0-9Ee+.]+\s*$"
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
        if _CP2K_FORCE_EVAL_HEADER.search(line):
            capture = True
            forces = []
            continue
        if "Atom   Kind   Element" in line:
            capture = True
            continue
        if capture:
            if not line.strip() or line.lstrip().startswith("SUM") or line.startswith("FORCES| Sum"):
                capture = False
                continue
            force_eval_match = _CP2K_FORCE_EVAL_LINE.match(line)
            if force_eval_match:
                forces.append(
                    [
                        float(force_eval_match.group(1)),
                        float(force_eval_match.group(2)),
                        float(force_eval_match.group(3)),
                    ]
                )
                continue
            match = _CP2K_FORCE_LINE.match(line)
            if match:
                forces.append([float(match.group(1)), float(match.group(2)), float(match.group(3))])
    if not forces:
        msg = f"No CP2K force block found in {path}"
        raise DataError(msg)
    return np.asarray(forces, dtype=float) * (Hartree / Bohr)
