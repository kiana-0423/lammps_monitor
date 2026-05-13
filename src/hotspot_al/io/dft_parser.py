"""Parsers for extracting DFT labels from external engine outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from hotspot_al.cp2k.cp2k_force_parser import parse_cp2k_forces as _parse_cp2k_forces


def parse_cp2k_forces(path: str | Path) -> np.ndarray:
    """Parse atomic forces from a CP2K output file into eV/Å."""

    return _parse_cp2k_forces(path)


def parse_forces(path: str | Path, engine: str = "cp2k") -> np.ndarray:
    """Parse forces from a supported DFT engine output."""

    normalized = engine.lower()
    if normalized == "cp2k":
        return parse_cp2k_forces(path)
    if normalized in {"gaussian", "orca"}:
        msg = f"{engine} force parsing is not implemented yet."
        raise NotImplementedError(msg)
    msg = f"Unsupported DFT engine: {engine}"
    raise ValueError(msg)
