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


def test_candidate_pool_can_disable_deduplication() -> None:
    deduplicated = CandidatePool(diversity_threshold=0.1, max_candidates=5)
    deduplicated.add(_region(), score=1.0)
    deduplicated.add(_region(), score=2.0)

    unfiltered = CandidatePool(diversity_threshold=0.1, max_candidates=5, deduplicate=False)
    unfiltered.add(_region(), score=1.0)
    unfiltered.add(_region(), score=2.0)

    assert len(deduplicated.select()) == 1
    assert [candidate.score for candidate in unfiltered.select()] == [2.0, 1.0]
