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
    force_blocks = _parse_force_eval_blocks(lines) or _parse_atomic_force_blocks(lines)
    forces = force_blocks[-1] if force_blocks else []
    if not forces:
        msg = f"No CP2K force block found in {path}"
        raise DataError(msg)
    return np.asarray(forces, dtype=float) * (Hartree / Bohr)


def _parse_force_eval_blocks(lines: list[str]) -> list[list[list[float]]]:
    blocks: list[list[list[float]]] = []
    current: list[list[float]] = []
    capture = False
    for line in lines:
        if _CP2K_FORCE_EVAL_HEADER.search(line):
            if current:
                blocks.append(current)
            current = []
            capture = True
            continue
        if not capture:
            continue
        if not line.strip() or line.startswith("FORCES| Sum"):
            if current:
                blocks.append(current)
                current = []
            capture = False
            continue
        match = _CP2K_FORCE_EVAL_LINE.match(line)
        if match:
            current.append([float(match.group(1)), float(match.group(2)), float(match.group(3))])
    if current:
        blocks.append(current)
    return blocks


def _parse_atomic_force_blocks(lines: list[str]) -> list[list[list[float]]]:
    blocks: list[list[list[float]]] = []
    current: list[list[float]] = []
    seen_header = False
    capture = False
    for line in lines:
        if _CP2K_FORCE_HEADER.search(line):
            if current:
                blocks.append(current)
            current = []
            seen_header = True
            capture = False
            continue
        if seen_header and "Atom   Kind   Element" in line:
            capture = True
            continue
        if not capture:
            continue
        if not line.strip() or line.lstrip().startswith("SUM"):
            if current:
                blocks.append(current)
                current = []
            capture = False
            seen_header = False
            continue
        match = _CP2K_FORCE_LINE.match(line)
        if match:
            current.append([float(match.group(1)), float(match.group(2)), float(match.group(3))])
    if current:
        blocks.append(current)
    return blocks
