"""Tests for cluster extraction and radial region assignment."""

from __future__ import annotations

from ase import Atoms

from hotspot_al.extraction.cluster_extractor import extract_cluster_region


def test_cluster_extraction_assigns_core_inner_and_boundary_regions() -> None:
    atoms = Atoms(
        symbols=["H", "H", "H", "H", "H"],
        positions=[
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [8.0, 0.0, 0.0],
        ],
        pbc=False,
    )
    config = {
        "extraction": {
            "core_radius": 1.5,
            "extract_radius": 5.5,
            "boundary_thickness": 1.5,
            "min_atoms": 1,
            "max_atoms": 10,
            "vacuum_padding": 2.0,
        }
    }

    region = extract_cluster_region(atoms, [1], config=config)
    assert region.original_indices == [0, 1, 2, 3]
    assert region.core_indices == [1]
    assert region.inner_buffer_indices == [0, 2]
    assert region.boundary_indices == [3]
    assert region.outer_buffer_indices == [3]
