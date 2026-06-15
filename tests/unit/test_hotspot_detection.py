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


def test_no_merge_when_far_apart() -> None:
    atoms = Atoms(
        symbols=["H", "H", "H", "H"],
        positions=[
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [6.5, 0.0, 0.0],
        ],
        pbc=False,
    )
    scores = np.array([7.0, 7.1, 8.0, 7.9])

    hotspots = detect_hotspots(atoms, scores, threshold=6.0, merge_radius=1.0)

    assert len(hotspots) == 2
    assert [hotspot.core_atom_indices for hotspot in hotspots] == [[2, 3], [0, 1]]


def test_no_hotspot_when_all_below_threshold() -> None:
    atoms = Atoms("H3", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], pbc=False)
    scores = np.array([1.0, 5.99, 2.0])

    hotspots = detect_hotspots(atoms, scores, threshold=6.0, merge_radius=1.0)

    assert hotspots == []


def test_single_atom_hotspot() -> None:
    atoms = Atoms("H3", positions=[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0]], pbc=False)
    scores = np.array([1.0, 6.2, 2.0])

    hotspots = detect_hotspots(atoms, scores, threshold=6.0, merge_radius=1.0)

    assert len(hotspots) == 1
    assert hotspots[0].core_atom_indices == [1]


def test_exact_threshold_boundary() -> None:
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [9.8, 0.0, 0.0]], cell=[10.0, 10.0, 10.0], pbc=True)
    scores = np.array([6.0, 6.1])

    hotspots = detect_hotspots(atoms, scores, threshold=6.0, merge_radius=0.5)

    assert len(hotspots) == 1
    assert hotspots[0].core_atom_indices == [0, 1]
