"""Tests for mask generation from extracted region labels."""

from __future__ import annotations

import numpy as np
from ase import Atoms

from hotspot_al.config import load_config
from hotspot_al.models import ExtractedRegion
from hotspot_al.training.mask_generator import generate_atom_mask


def test_mask_generator_matches_default_config_weights() -> None:
    region = ExtractedRegion(
        atoms=Atoms(symbols=["H", "H", "H", "H", "H"], positions=np.zeros((5, 3))),
        original_indices=[0, 1, 2, 3, -1],
        core_indices=[0],
        inner_buffer_indices=[1],
        outer_buffer_indices=[2],
        boundary_indices=[3],
        h_cap_indices=[4],
        hotspot_indices=[0],
        metadata={},
    )
    mask = generate_atom_mask(region, load_config())
    assert np.allclose(mask, np.array([1.0, 0.3, 0.0, 0.0, 0.0]))


def test_mask_generator_custom_weights() -> None:
    region = ExtractedRegion(
        atoms=Atoms(symbols=["H", "H", "H"], positions=np.zeros((3, 3))),
        original_indices=[0, 1, 2],
        core_indices=[0],
        inner_buffer_indices=[1],
        outer_buffer_indices=[],
        boundary_indices=[2],
        h_cap_indices=[],
        hotspot_indices=[0],
    )

    mask = generate_atom_mask(region, {"training_mask": {"core": 1.0, "inner_buffer": 0.5, "boundary": 0.0}})

    assert np.allclose(mask, np.array([1.0, 0.5, 0.0]))


def test_mask_generator_all_core_returns_ones() -> None:
    region = ExtractedRegion(
        atoms=Atoms(symbols=["H", "H", "H"], positions=np.zeros((3, 3))),
        original_indices=[0, 1, 2],
        core_indices=[0, 1, 2],
        inner_buffer_indices=[],
        outer_buffer_indices=[],
        boundary_indices=[],
        h_cap_indices=[],
        hotspot_indices=[0],
    )

    mask = generate_atom_mask(region, load_config())

    assert np.allclose(mask, np.ones(3))


def test_mask_generator_unknown_label_defaults_to_zero() -> None:
    region = ExtractedRegion(
        atoms=Atoms(symbols=["H"], positions=np.zeros((1, 3))),
        original_indices=[0],
        core_indices=[],
        inner_buffer_indices=[],
        outer_buffer_indices=[],
        boundary_indices=[],
        h_cap_indices=[],
        hotspot_indices=[],
        region_labels=["unknown"],
    )

    mask = generate_atom_mask(region, load_config())

    assert np.allclose(mask, np.array([0.0]))


def test_mask_generator_mask_length_matches_atoms() -> None:
    region = ExtractedRegion(
        atoms=Atoms(symbols=["H", "H", "H", "H"], positions=np.zeros((4, 3))),
        original_indices=[0, 1, 2, 3],
        core_indices=[0],
        inner_buffer_indices=[1],
        outer_buffer_indices=[2],
        boundary_indices=[3],
        h_cap_indices=[],
        hotspot_indices=[0],
    )

    mask = generate_atom_mask(region, load_config())

    assert len(mask) == len(region.atoms)
