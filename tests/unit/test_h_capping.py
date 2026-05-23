"""Tests for conservative hydrogen capping on broken covalent bonds."""

from __future__ import annotations

import numpy as np
from ase import Atoms

from hotspot_al.config import load_config
from hotspot_al.extraction.h_capping import add_h_caps
from hotspot_al.models import ExtractedRegion
from hotspot_al.training.mask_generator import generate_atom_mask


def test_ethane_boundary_carbon_gets_one_h_cap() -> None:
    original = Atoms(
        symbols=["C", "C", "H", "H", "H", "H", "H", "H"],
        positions=[
            [0.00, 0.00, 0.00],
            [1.54, 0.00, 0.00],
            [-0.63, 0.90, 0.00],
            [-0.63, -0.45, 0.78],
            [-0.63, -0.45, -0.78],
            [2.17, 0.90, 0.00],
            [2.17, -0.45, 0.78],
            [2.17, -0.45, -0.78],
        ],
        pbc=False,
    )

    region_atoms = original[[0, 2, 3, 4]].copy()
    region = ExtractedRegion(
        atoms=region_atoms,
        original_indices=[0, 2, 3, 4],
        core_indices=[],
        inner_buffer_indices=[],
        outer_buffer_indices=[0],
        boundary_indices=[0],
        h_cap_indices=[],
        hotspot_indices=[0],
        metadata={},
    )

    updated = add_h_caps(original, region, config=load_config())
    assert len(updated.h_cap_indices) == 1

    h_index = updated.h_cap_indices[0]
    cap_vector = updated.atoms.positions[h_index] - updated.atoms.positions[0]
    cap_vector /= np.linalg.norm(cap_vector)
    expected = np.array([1.0, 0.0, 0.0])
    assert np.dot(cap_vector, expected) > 0.99

    mask = generate_atom_mask(updated, load_config())
    assert mask[h_index] == 0.0
