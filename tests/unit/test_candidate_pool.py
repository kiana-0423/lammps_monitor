"""Tests for active-learning candidate pool selection."""

from __future__ import annotations

import numpy as np
from ase import Atoms

from hotspot_al.active_learning.candidate_pool import CandidatePool
from hotspot_al.active_learning.deduplication import (
    CandidateFingerprint,
    deduplicate_candidates,
    pair_distance_histogram,
)
from hotspot_al.models import ExtractedRegion


def _region() -> ExtractedRegion:
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
    return ExtractedRegion(
        atoms=atoms,
        original_indices=[0, 1],
        core_indices=[0],
        inner_buffer_indices=[1],
        outer_buffer_indices=[],
        boundary_indices=[],
        h_cap_indices=[],
        hotspot_indices=[0],
    )


def _h2_region(bond_length: float, *, score: float | None = None) -> ExtractedRegion:
    region = _region()
    region.atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [bond_length, 0.0, 0.0]])
    if score is not None:
        region.metadata["score"] = score
    return region


def test_candidate_pool_can_disable_deduplication() -> None:
    deduplicated = CandidatePool(diversity_threshold=0.1, max_candidates=5)
    deduplicated.add(_region(), score=1.0)
    deduplicated.add(_region(), score=2.0)

    unfiltered = CandidatePool(diversity_threshold=0.1, max_candidates=5, deduplicate=False)
    unfiltered.add(_region(), score=1.0)
    unfiltered.add(_region(), score=2.0)

    assert len(deduplicated.select()) == 1
    assert [candidate.score for candidate in unfiltered.select()] == [2.0, 1.0]


def test_candidate_pool_select_top_by_score() -> None:
    pool = CandidatePool(diversity_threshold=0.0, max_candidates=2, deduplicate=False)
    pool.add(_h2_region(0.72, score=0.2), score=0.2, metadata={"score": 0.2})
    pool.add(_h2_region(0.74, score=2.0), score=2.0, metadata={"score": 2.0})
    pool.add(_h2_region(0.76, score=1.1), score=1.1, metadata={"score": 1.1})

    selected = pool.select()

    assert [candidate.metadata["score"] for candidate in selected] == [2.0, 1.1]
    assert len(selected) == 2


def test_candidate_pool_deduplicate_removes_similar() -> None:
    pool = CandidatePool(diversity_threshold=0.01, max_candidates=5)
    pool.add(_h2_region(0.740), score=1.0)
    pool.add(_h2_region(0.741), score=2.0)

    selected = pool.select()

    assert len(selected) == 1
    assert selected[0].score == 2.0


def test_candidate_pool_empty_pool_returns_empty() -> None:
    assert CandidatePool().select() == []


def test_candidate_pool_max_candidates_larger_than_pool() -> None:
    pool = CandidatePool(diversity_threshold=0.0, max_candidates=10, deduplicate=False)
    for score in (0.5, 1.5, 1.0):
        pool.add(_h2_region(0.7 + score), score=score)

    selected = pool.select()

    assert len(selected) == 3
    assert [candidate.score for candidate in selected] == [1.5, 1.0, 0.5]


def test_pair_distance_histogram_matches_reference_loop() -> None:
    region = ExtractedRegion(
        atoms=Atoms("H4", positions=[[0.0, 0.0, 0.0], [0.7, 0.0, 0.0], [0.0, 1.2, 0.0], [0.0, 0.0, 1.8]]),
        original_indices=[0, 1, 2, 3],
        core_indices=[0],
        inner_buffer_indices=[1, 2, 3],
        outer_buffer_indices=[],
        boundary_indices=[],
        h_cap_indices=[],
        hotspot_indices=[0],
    )
    positions = region.atoms.get_positions()
    distances = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            distances.append(float(np.linalg.norm(positions[i] - positions[j])))
    expected, _ = np.histogram(distances, bins=16, range=(0.0, 8.0), density=True)

    assert np.allclose(pair_distance_histogram(region), expected)


def test_pair_distance_histogram_empty_or_single_region_returns_zero() -> None:
    empty = ExtractedRegion(Atoms(), [], [], [], [], [], [], [])
    single = ExtractedRegion(Atoms("H", positions=[[0.0, 0.0, 0.0]]), [0], [0], [], [], [], [], [0])

    assert np.allclose(pair_distance_histogram(empty), np.zeros(16))
    assert np.allclose(pair_distance_histogram(single), np.zeros(16))


def test_deduplicate_candidates_keeps_higher_score_for_identical_fingerprints() -> None:
    fingerprint = np.array([1.0, 0.0, 0.0])
    low = CandidateFingerprint(_region(), score=1.0, fingerprint=fingerprint, metadata={"name": "low"})
    high = CandidateFingerprint(_region(), score=2.0, fingerprint=fingerprint.copy(), metadata={"name": "high"})

    selected = deduplicate_candidates([low, high], diversity_threshold=0.1)

    assert len(selected) == 1
    assert selected[0].metadata["name"] == "high"


def test_deduplicate_candidates_threshold_boundary_keeps_candidate() -> None:
    left = CandidateFingerprint(_region(), score=2.0, fingerprint=np.array([0.0]), metadata={"name": "left"})
    right = CandidateFingerprint(_region(), score=1.0, fingerprint=np.array([0.1]), metadata={"name": "right"})

    selected = deduplicate_candidates([left, right], diversity_threshold=0.1)

    assert [candidate.metadata["name"] for candidate in selected] == ["left", "right"]
