"""Tests for slab patch extraction."""

from __future__ import annotations

import numpy as np
from ase import Atoms

from hotspot_al.extraction.slab_extractor import extract_slab_patch


def test_slab_patch_from_triclinic_cell_uses_local_orthorhombic_patch_cell() -> None:
    atoms = Atoms(
        "H5",
        positions=[
            [0.0, 0.0, 0.0],
            [1.0, 0.2, 0.1],
            [2.0, 0.1, -0.1],
            [8.0, 6.0, 0.0],
            [9.0, 7.0, 0.0],
        ],
        cell=np.array([[10.0, 2.0, 0.5], [0.0, 8.0, 1.0], [0.0, 0.0, 12.0]]),
        pbc=[True, True, False],
    )
    config = {
        "extraction": {
            "core_radius": 1.5,
            "extract_radius": 4.0,
            "boundary_thickness": 1.0,
        }
    }

    region = extract_slab_patch(atoms, [1], config=config, xy_lengths=(5.0, 5.0), z_thickness=4.0)

    assert np.allclose(region.atoms.cell.array[:2, :2], np.diag([5.0, 5.0])[:2, :2])
    assert np.allclose(region.atoms.cell.array[np.triu_indices(3, k=1)], 0.0)
    assert region.atoms.pbc.tolist() == [True, True, False]
    assert region.hotspot_indices == [region.original_indices.index(1)]
