"""Tests for LAMMPS dump parsing into FrameData."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hotspot_al.exceptions import DataError
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


def test_lammps_dump_reader_converts_scaled_coordinates_without_origin_shift(tmp_path: Path) -> None:
    path = tmp_path / "scaled.lammpstrj"
    path.write_text(
        "\n".join(
            [
                "ITEM: TIMESTEP",
                "0",
                "ITEM: NUMBER OF ATOMS",
                "2",
                "ITEM: BOX BOUNDS pp pp pp",
                "10 20",
                "-5 5",
                "2 12",
                "ITEM: ATOMS id type xs ys zs",
                "1 1 0.25 0.50 0.75",
                "2 1 0.00 0.00 0.00",
            ]
        ),
        encoding="utf-8",
    )

    frame = read_dump(path, type_map={1: "H"})[0]

    assert np.allclose(frame.atoms.cell.array, np.diag([10.0, 10.0, 10.0]))
    assert np.allclose(frame.atoms.positions[0], [2.5, 5.0, 7.5])
    assert np.allclose(frame.atoms.positions[1], [0.0, 0.0, 0.0])


def test_truncated_dump_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "truncated.lammpstrj"
    path.write_text(
        "\n".join(
            [
                "ITEM: TIMESTEP",
                "0",
                "ITEM: NUMBER OF ATOMS",
                "2",
                "ITEM: BOX BOUNDS pp pp pp",
                "0 10",
                "0 10",
                "0 10",
                "ITEM: ATOMS id type x y z",
                "1 1 0.0 0.0 0.0",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(DataError, match="truncated atom section"):
        read_dump(path, type_map={1: "H"})
