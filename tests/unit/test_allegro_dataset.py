"""Tests for Allegro extxyz export with mask weights."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import read

from hotspot_al.config import load_config
from hotspot_al.models import ExtractedRegion
from hotspot_al.training.allegro_adapter import write_allegro_dataset


def _region_with_all_labels() -> ExtractedRegion:
    return ExtractedRegion(
        atoms=Atoms(
            symbols=["C", "H", "H", "H", "H"],
            positions=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                ]
            ),
            cell=np.diag([8.0, 8.0, 8.0]),
            pbc=False,
        ),
        original_indices=[0, 1, 2, 3, -1],
        core_indices=[0],
        inner_buffer_indices=[1],
        outer_buffer_indices=[2],
        boundary_indices=[3],
        h_cap_indices=[4],
        hotspot_indices=[0],
        region_labels=["core", "inner_buffer", "outer_buffer", "boundary", "h_cap"],
        metadata={"event_id": "evt-2"},
    )


def test_allegro_adapter_writes_extxyz_with_expected_fields(tmp_path: Path) -> None:
    region = _region_with_all_labels()
    forces = np.arange(len(region.atoms) * 3, dtype=float).reshape(len(region.atoms), 3)

    written = write_allegro_dataset(region, forces=forces, output_dir=tmp_path, config=load_config())
    atoms = read(written["dataset"], format="extxyz")
    text = written["dataset"].read_text(encoding="utf-8")

    assert "forces" in text
    assert "mask_weights" in atoms.arrays
    assert "region_code" in atoms.arrays
    assert len(atoms.arrays["mask_weights"]) == len(atoms)
    assert np.allclose(atoms.get_forces(), forces)
    assert np.allclose(atoms.arrays["mask_weights"], np.array([1.0, 0.3, 0.0, 0.0, 0.0]))
    assert np.array_equal(atoms.arrays["region_code"], np.array([0, 1, 2, 3, 4]))


def test_allegro_adapter_rejects_force_shape_mismatch(tmp_path: Path) -> None:
    region = _region_with_all_labels()
    forces = np.zeros((len(region.atoms) - 1, 3))

    with pytest.raises(ValueError):
        write_allegro_dataset(region, forces=forces, output_dir=tmp_path, config=load_config())
