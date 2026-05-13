"""Tests for CP2K force parsing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase.units import Bohr, Hartree

from hotspot_al.cp2k.cp2k_force_parser import parse_cp2k_forces


def test_cp2k_force_parser_reads_mock_output(tmp_path: Path) -> None:
    output = """ ATOMIC FORCES in [a.u.]

 #  Atom   Kind   Element          X              Y              Z
      1      1      C        1.0000000000   0.0000000000  -1.0000000000
      2      1      H        0.5000000000   0.2500000000   0.1250000000
 SUM OF ATOMIC FORCES          0.0            0.0            0.0
"""
    path = tmp_path / "cp2k.out"
    path.write_text(output, encoding="utf-8")

    forces = parse_cp2k_forces(path)
    conversion = Hartree / Bohr
    expected = np.array([[1.0, 0.0, -1.0], [0.5, 0.25, 0.125]]) * conversion
    assert np.allclose(forces, expected)
