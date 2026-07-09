"""Tests for generic masked dataset entry writing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read

from hotspot_al.models import ExtractedRegion
from hotspot_al.training.dataset_writer import write_dataset_entry


def _region() -> ExtractedRegion:
    return ExtractedRegion(
        atoms=Atoms(
            "CHH",
            positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            cell=np.diag([8.0, 8.0, 8.0]),
            pbc=False,
        ),
        original_indices=[4, 5, -1],
        core_indices=[0],
        inner_buffer_indices=[1],
        outer_buffer_indices=[],
        boundary_indices=[],
        h_cap_indices=[2],
        hotspot_indices=[0],
        region_labels=["core", "inner_buffer", "h_cap"],
        metadata={"original_frame_id": 12, "hotspot_id": "evt-12"},
    )


def test_write_dataset_entry_writes_structure_labels_and_metadata(tmp_path: Path) -> None:
    region = _region()
    forces = np.arange(9, dtype=float).reshape(3, 3)
    mask = np.array([1.0, 0.3, 0.0])

    written = write_dataset_entry(
        region,
        forces=forces,
        mask=mask,
        output_dir=tmp_path,
        prefix="sample",
        extra_metadata={"energy_weight": 0.5, "cp2k_output": "cp2k.out"},
    )

    atoms = read(written["structure"], format="extxyz")
    labels = np.load(written["labels"])
    metadata = json.loads(written["metadata"].read_text(encoding="utf-8"))

    assert set(written) == {"structure", "labels", "metadata"}
    assert np.allclose(atoms.get_forces(), forces)
    assert np.allclose(atoms.arrays["mask_weights"], mask)
    assert np.array_equal(atoms.arrays["region_code"], np.array([0, 1, 4]))
    assert np.array_equal(labels["original_indices"], np.array([4, 5, -1]))
    assert np.array_equal(labels["h_cap_indices"], np.array([2]))
    assert metadata["region_labels"] == ["core", "inner_buffer", "h_cap"]
    assert metadata["masked_atom_indices"] == [2]
    assert metadata["extra_metadata"]["cp2k_output"] == "cp2k.out"
