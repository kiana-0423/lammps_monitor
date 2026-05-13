"""Compatibility wrapper around the CP2K force parser."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hotspot_al.cp2k.cp2k_force_parser import parse_cp2k_forces


def read_cp2k_forces(path: str | Path) -> np.ndarray:
    """Read CP2K atomic forces from an output file."""

    return parse_cp2k_forces(path)
