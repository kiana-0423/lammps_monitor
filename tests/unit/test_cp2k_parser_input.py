"""Tests for CP2K force parsing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.units import Bohr, Hartree

from hotspot_al.config import load_config
from hotspot_al.cp2k.cp2k_input import write_cp2k_inputs
from hotspot_al.cp2k.cp2k_force_parser import parse_cp2k_forces
from hotspot_al.exceptions import DataError
from hotspot_al.models import ExtractedRegion


def _toy_region() -> ExtractedRegion:
    return ExtractedRegion(
        atoms=Atoms(
            symbols=["C", "H"],
            positions=[[0.0, 0.0, 0.0], [1.09, 0.0, 0.0]],
            cell=np.diag([8.0, 8.0, 8.0]),
            pbc=False,
        ),
        original_indices=[0, 1],
        core_indices=[0],
        inner_buffer_indices=[1],
        outer_buffer_indices=[],
        boundary_indices=[],
        h_cap_indices=[],
        hotspot_indices=[0],
        region_labels=["core", "inner_buffer"],
        metadata={},
    )


def test_cp2k_force_parser_reads_toy_output(fixtures_dir: Path) -> None:
    path = fixtures_dir / "cp2k" / "toy_forces.out"

    forces = parse_cp2k_forces(path)
    conversion = Hartree / Bohr
    expected = np.array([[0.1, 0.0, -0.1], [0.05, 0.025, 0.0125]]) * conversion
    assert forces.shape == (2, 3)
    assert np.allclose(forces, expected)


def test_cp2k_force_parser_reads_force_eval_print(tmp_path: Path) -> None:
    path = tmp_path / "cp2k.out"
    path.write_text(
        "\n".join(
            [
                "FORCES| Atomic forces [hartree/bohr]",
                "FORCES|   Atom     x               y               z               |f|",
                "FORCES|      1 -1.00000000E-01  2.00000000E-01 -3.00000000E-01   3.74165739E-01",
                "FORCES|      2  4.00000000E-01 -5.00000000E-01  6.00000000E-01   8.77496439E-01",
                "FORCES| Sum     3.00000000E-01 -3.00000000E-01  3.00000000E-01",
            ]
        ),
        encoding="utf-8",
    )

    conversion = Hartree / Bohr
    expected = np.array([[-0.1, 0.2, -0.3], [0.4, -0.5, 0.6]]) * conversion
    assert np.allclose(parse_cp2k_forces(path), expected)


def test_cp2k_input_generator_writes_required_sections(tmp_path: Path) -> None:
    written = write_cp2k_inputs(_toy_region(), tmp_path, config=load_config(), job_name="toy")

    assert "single_point_input" in written
    text = written["single_point_input"].read_text(encoding="utf-8")
    assert "&CELL" in text
    assert "A 8.00000000 0.00000000 0.00000000" in text
    assert "&COORD" in text
    assert "METHOD Quickstep" in text
    assert "BASIS_SET DZVP-MOLOPT-SR-GTH" in text
    assert "POTENTIAL GTH-PBE" in text
    assert "&XC_FUNCTIONAL PBE" in text
    assert "  &PRINT\n    &FORCES MEDIUM" in text
    assert "    &PRINT\n      &FORCES" not in text


def test_parse_empty_force_section_raises(tmp_path: Path) -> None:
    path = tmp_path / "cp2k.out"
    path.write_text("CP2K output without an ATOMIC FORCES section\n", encoding="utf-8")

    with pytest.raises(DataError, match="No CP2K force block"):
        parse_cp2k_forces(path)
