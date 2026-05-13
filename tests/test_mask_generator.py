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
