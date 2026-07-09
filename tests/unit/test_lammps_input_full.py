"""Tests for complete LAMMPS input generation."""

from __future__ import annotations

from pathlib import Path

from ase import Atoms

from hotspot_al.config import load_config
from hotspot_al.lammps.lammps_controller import LAMMPSController
from hotspot_al.lammps.lammps_input import build_full_lammps_input, write_full_lammps_input

PAIR_BLOCK = "pair_style allegro model.pth\npair_coeff * *"


def test_build_full_lammps_input_supports_restart() -> None:
    text = build_full_lammps_input(pair_style_block=PAIR_BLOCK, config=load_config(), restart_file="old.restart")

    assert "read_restart old.restart" in text
    assert "pair_style allegro model.pth" in text
    assert "dump online_dump all custom" in text


def test_write_full_lammps_input_writes_data_file(tmp_path: Path) -> None:
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]], cell=[5.0, 5.0, 5.0], pbc=True)
    input_path = write_full_lammps_input(
        tmp_path / "in.hotspot_al",
        pair_style_block=PAIR_BLOCK,
        config=load_config(),
        atoms=atoms,
        data_file="system.data",
    )

    assert input_path.is_file()
    assert (tmp_path / "system.data").is_file()
    assert "read_data system.data" in input_path.read_text(encoding="utf-8")


def test_lammps_controller_from_atoms_prepares_run_directory(tmp_path: Path) -> None:
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=[5.0, 5.0, 5.0], pbc=True)

    controller = LAMMPSController.from_atoms(atoms, pair_style_block=PAIR_BLOCK, config=load_config(), work_dir=tmp_path)

    assert controller.input_file == tmp_path / "in.hotspot_al"
    assert controller.dump_file == tmp_path / "dump.online.lammpstrj"
    assert controller.input_file.is_file()
