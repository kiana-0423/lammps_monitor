"""Tests for active-learning candidate pool selection."""

from __future__ import annotations

from ase import Atoms

from hotspot_al.active_learning.candidate_pool import CandidatePool
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
