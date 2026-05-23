"""Tests for LAMMPS dump parsing into FrameData."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hotspot_al.io.lammps_reader import read_dump


def test_lammps_dump_reader_parses_positions_forces_cell_and_step(fixtures_dir: Path) -> None:
    path = fixtures_dir / "lammps" / "toy_hotspot.lammpstrj"

    frames = read_dump(path, type_map={1: "C", 3: "H"}, timestep_fs=0.5)
    assert len(frames) == 1

    frame = frames[0]
    assert frame.step == 100
    assert frame.time == 50.0
    assert len(frame.atoms) == 8
    assert frame.atoms.get_chemical_symbols() == ["C", "C", "H", "H", "H", "H", "H", "H"]
    assert frame.forces is not None
    assert frame.forces.shape == (8, 3)
    assert np.allclose(frame.atoms.get_positions()[0], np.array([4.0, 6.0, 6.0]))
    assert np.allclose(frame.forces[0], np.array([9.0, 0.0, 0.0]))
    assert np.allclose(frame.atoms.cell.array, np.diag([12.0, 12.0, 12.0]))
