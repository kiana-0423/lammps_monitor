"""Tests for Allegro extxyz export with mask weights."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read

from hotspot_al.config import load_config
from hotspot_al.models import ExtractedRegion
from hotspot_al.training.allegro_adapter import write_allegro_dataset


def test_allegro_adapter_writes_extxyz_with_mask_field(tmp_path: Path) -> None:
    region = ExtractedRegion(
        atoms=Atoms(symbols=["O", "H"], positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0]], cell=np.diag([8.0, 8.0, 8.0]), pbc=False),
        original_indices=[0, 1],
        core_indices=[0],
        inner_buffer_indices=[1],
        outer_buffer_indices=[],
        boundary_indices=[],
        h_cap_indices=[],
        hotspot_indices=[0],
        region_labels=["core", "inner_buffer"],
        metadata={"event_id": "evt-2"},
    )
    forces = np.array([[0.2, 0.0, 0.0], [-0.2, 0.0, 0.0]])

    written = write_allegro_dataset(region, forces=forces, output_dir=tmp_path, config=load_config())
    atoms = read(written["dataset"], format="extxyz")
    assert "mask_weights" in atoms.arrays
    assert np.allclose(atoms.arrays["mask_weights"], np.array([1.0, 0.3]))
