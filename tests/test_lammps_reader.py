"""Tests for LAMMPS dump parsing into FrameData."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hotspot_al.io.lammps_reader import read_dump


def test_lammps_dump_reader_parses_positions_forces_cell_and_step(tmp_path: Path) -> None:
    dump_text = """ITEM: TIMESTEP
100
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 12.0
0.0 14.0
ITEM: ATOMS id type x y z fx fy fz
2 2 2.0 3.0 4.0 -0.1 0.0 0.1
1 1 1.0 2.0 3.0 0.1 0.2 0.3
"""
    path = tmp_path / "dump.lammpstrj"
    path.write_text(dump_text, encoding="utf-8")

    frames = read_dump(path, type_map={1: "C", 2: "O"}, timestep_fs=0.5)
    assert len(frames) == 1

    frame = frames[0]
    assert frame.step == 100
    assert frame.time == 50.0
    assert frame.atoms.get_chemical_symbols() == ["C", "O"]
    assert np.allclose(frame.atoms.get_positions(), np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]))
    assert np.allclose(frame.forces, np.array([[0.1, 0.2, 0.3], [-0.1, 0.0, 0.1]]))
    assert np.allclose(frame.atoms.cell.array, np.diag([10.0, 12.0, 14.0]))
