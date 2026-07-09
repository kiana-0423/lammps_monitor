"""Tests for radial extraction region assignment."""

from __future__ import annotations

import numpy as np

from hotspot_al.extraction.boundary_regions import assign_radial_regions, region_label_vector


def test_assign_radial_regions_keeps_outer_buffer_and_boundary_distinct() -> None:
    regions = assign_radial_regions(
        np.array([0.0, 2.0, 3.2, 4.2, 5.0]),
        core_radius=1.0,
        extract_radius=5.0,
        boundary_thickness=1.0,
    )

    assert regions["core"] == [0]
    assert regions["inner_buffer"] == [1]
    assert regions["outer_buffer"] == [2]
    assert regions["boundary"] == [3, 4]
    assert set(regions["outer_buffer"]).isdisjoint(regions["boundary"])


def test_region_label_vector_preserves_all_radial_roles() -> None:
    labels = region_label_vector(
        4,
        {
            "core": [0],
            "inner_buffer": [1],
            "outer_buffer": [2],
            "boundary": [3],
        },
    )

    assert labels == ["core", "inner_buffer", "outer_buffer", "boundary"]
