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


def test_benzene_boundary_carbon_gets_h_cap() -> None:
    original = Atoms(
        symbols=["C"] * 6 + ["H"] * 6,
        positions=[
            [1.40, 0.00, 0.00],
            [0.70, 1.21, 0.00],
            [-0.70, 1.21, 0.00],
            [-1.40, 0.00, 0.00],
            [-0.70, -1.21, 0.00],
            [0.70, -1.21, 0.00],
            [2.49, 0.00, 0.00],
            [1.24, 2.16, 0.00],
            [-1.24, 2.16, 0.00],
            [-2.49, 0.00, 0.00],
            [-1.24, -2.16, 0.00],
            [1.24, -2.16, 0.00],
        ],
        pbc=False,
    )
    region = ExtractedRegion(
        atoms=original[[0, 1, 6]].copy(),
        original_indices=[0, 1, 6],
        core_indices=[],
        inner_buffer_indices=[],
        outer_buffer_indices=[0],
        boundary_indices=[0],
        h_cap_indices=[],
        hotspot_indices=[0],
    )

    updated = add_h_caps(original, region, config=load_config())

    assert len(updated.h_cap_indices) == 1
    assert updated.metadata["h_caps"][0]["broken_partner_original_index"] == 5


def test_double_bond_carbon_gets_two_h_caps() -> None:
    original = Atoms(
        symbols=["C", "C", "H", "H", "H", "H"],
        positions=[
            [0.00, 0.00, 0.00],
            [1.34, 0.00, 0.00],
            [-0.55, 0.93, 0.00],
            [-0.55, -0.93, 0.00],
            [1.89, 0.93, 0.00],
            [1.89, -0.93, 0.00],
        ],
        pbc=False,
    )
    region = ExtractedRegion(
        atoms=original[[0, 1]].copy(),
        original_indices=[0, 1],
        core_indices=[],
        inner_buffer_indices=[],
        outer_buffer_indices=[0],
        boundary_indices=[0],
        h_cap_indices=[],
        hotspot_indices=[0],
    )

    updated = add_h_caps(original, region, config=load_config())

    assert len(updated.h_cap_indices) == 2
    assert {item["broken_partner_original_index"] for item in updated.metadata["h_caps"]} == {2, 3}


def test_no_h_cap_when_element_not_in_bond_lengths() -> None:
    original = Atoms("FH", positions=[[0.0, 0.0, 0.0], [0.92, 0.0, 0.0]], pbc=False)
    region = ExtractedRegion(
        atoms=original[[0]].copy(),
        original_indices=[0],
        core_indices=[],
        inner_buffer_indices=[],
        outer_buffer_indices=[0],
        boundary_indices=[0],
        h_cap_indices=[],
        hotspot_indices=[0],
    )

    updated = add_h_caps(original, region, config=load_config())

    assert updated.h_cap_indices == []


def test_oxide_skipped_by_default() -> None:
    original = Atoms("OSiO", positions=[[-1.6, 0.0, 0.0], [0.0, 0.0, 0.0], [1.6, 0.0, 0.0]], pbc=False)
    region = ExtractedRegion(
        atoms=original[[1]].copy(),
        original_indices=[1],
        core_indices=[],
        inner_buffer_indices=[],
        outer_buffer_indices=[0],
        boundary_indices=[0],
        h_cap_indices=[],
        hotspot_indices=[0],
    )

    updated = add_h_caps(original, region, config=load_config())

    assert updated.h_cap_indices == []


def test_h_cap_position_along_bond_vector() -> None:
    original = Atoms("CH", positions=[[0.0, 0.0, 0.0], [0.9, 0.5, 0.0]], pbc=False)
    region = ExtractedRegion(
        atoms=original[[0]].copy(),
        original_indices=[0],
        core_indices=[],
        inner_buffer_indices=[],
        outer_buffer_indices=[0],
        boundary_indices=[0],
        h_cap_indices=[],
        hotspot_indices=[0],
    )

    updated = add_h_caps(original, region, config=load_config())

    h_index = updated.h_cap_indices[0]
    cap_vector = updated.atoms.positions[h_index] - updated.atoms.positions[0]
    broken_bond_vector = original.positions[1] - original.positions[0]
    cap_vector /= np.linalg.norm(cap_vector)
    broken_bond_vector /= np.linalg.norm(broken_bond_vector)
    assert np.dot(cap_vector, broken_bond_vector) > 0.999
    assert np.isclose(np.linalg.norm(updated.atoms.positions[h_index] - updated.atoms.positions[0]), 1.09)


def test_no_broken_bonds_no_h_caps() -> None:
    original = Atoms("CH", positions=[[0.0, 0.0, 0.0], [1.09, 0.0, 0.0]], pbc=False)
    region = ExtractedRegion(
        atoms=original.copy(),
        original_indices=[0, 1],
        core_indices=[],
        inner_buffer_indices=[],
        outer_buffer_indices=[0],
        boundary_indices=[0],
        h_cap_indices=[],
        hotspot_indices=[0],
    )

    updated = add_h_caps(original, region, config=load_config())

    assert updated.h_cap_indices == []
