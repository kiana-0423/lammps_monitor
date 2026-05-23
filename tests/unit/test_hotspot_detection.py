"""Tests for hotspot detection and spatial merging."""

from __future__ import annotations

import numpy as np
from ase import Atoms

from hotspot_al.hotspot.hotspot_detector import detect_hotspots


def test_hotspot_detection_merges_nearby_anomalous_atoms() -> None:
    atoms = Atoms(
        symbols=["H", "H", "H", "H"],
        positions=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [7.5, 0.0, 0.0],
            [20.0, 0.0, 0.0],
        ],
        pbc=False,
    )
    scores = np.array([7.2, 6.4, 6.1, 1.0])
    hotspots = detect_hotspots(atoms, scores, threshold=6.0, merge_radius=2.0, step=12, trigger_reasons=["force_burst"])

    assert len(hotspots) == 2
    assert hotspots[0].core_atom_indices == [0, 1]
    assert hotspots[1].core_atom_indices == [2]
    assert hotspots[0].step == 12
    assert hotspots[0].trigger_reasons == ["force_burst"]
