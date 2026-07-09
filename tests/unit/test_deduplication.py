"""Focused tests for active-learning candidate deduplication."""

from __future__ import annotations

import numpy as np
from ase import Atoms

from hotspot_al.active_learning.deduplication import (
    CandidateFingerprint,
    deduplicate_candidates,
    fingerprint_distance,
    fingerprint_region,
    pair_distance_histogram,
    type_weighted_pair_distance_histogram,
)
from hotspot_al.models import ExtractedRegion


def _region(symbols: str = "H2", positions: list[list[float]] | None = None) -> ExtractedRegion:
    atoms = Atoms(symbols, positions=positions if positions is not None else [[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
    return ExtractedRegion(
        atoms=atoms,
        original_indices=list(range(len(atoms))),
        core_indices=[0] if len(atoms) else [],
        inner_buffer_indices=list(range(1, len(atoms))),
        outer_buffer_indices=[],
        boundary_indices=[],
        h_cap_indices=[],
        hotspot_indices=[0] if len(atoms) else [],
    )


def _candidate(name: str, score: float, fingerprint: np.ndarray) -> CandidateFingerprint:
    return CandidateFingerprint(_region(), score=score, fingerprint=fingerprint, metadata={"name": name})


def test_empty_candidates() -> None:
    assert deduplicate_candidates([], diversity_threshold=0.1) == []


def test_single_candidate() -> None:
    candidate = _candidate("only", 1.0, np.array([0.0, 1.0]))

    assert deduplicate_candidates([candidate], diversity_threshold=0.1) == [candidate]


def test_identical_fingerprints_keep_high_score() -> None:
    selected = deduplicate_candidates(
        [
            _candidate("low", 1.0, np.array([1.0, 0.0])),
            _candidate("high", 2.0, np.array([1.0, 0.0])),
        ],
        diversity_threshold=0.1,
    )

    assert [candidate.metadata["name"] for candidate in selected] == ["high"]


def test_threshold_boundary_is_kept() -> None:
    selected = deduplicate_candidates(
        [
            _candidate("left", 2.0, np.array([0.0])),
            _candidate("right", 1.0, np.array([0.1])),
        ],
        diversity_threshold=0.1,
    )

    assert [candidate.metadata["name"] for candidate in selected] == ["left", "right"]


def test_fingerprint_distance() -> None:
    assert fingerprint_distance(np.array([0.0, 0.0]), np.array([3.0, 4.0])) == 5.0


def test_pair_distance_histogram_empty_region() -> None:
    histogram = pair_distance_histogram(_region("", []))

    assert np.allclose(histogram, np.zeros(16))


def test_pair_distance_histogram_single_atom_region() -> None:
    histogram = pair_distance_histogram(_region("H", [[0.0, 0.0, 0.0]]))

    assert np.allclose(histogram, np.zeros(16))


def test_pair_distance_histogram_small_region() -> None:
    histogram = pair_distance_histogram(_region("H2", [[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]]), bins=8, cutoff=4.0)

    assert histogram.shape == (8,)
    assert np.count_nonzero(histogram) == 1


def test_type_weighted_pair_distance_histogram_distinguishes_atom_types() -> None:
    h2 = _region("H2", [[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
    co = _region("CO", [[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])

    h2_fingerprint = type_weighted_pair_distance_histogram(h2, bins=8, cutoff=4.0)
    co_fingerprint = type_weighted_pair_distance_histogram(co, bins=8, cutoff=4.0)

    assert h2_fingerprint.shape == (24,)
    assert not np.allclose(h2_fingerprint, co_fingerprint)


def test_fingerprint_region_rejects_unknown_mode() -> None:
    try:
        fingerprint_region(_region(), mode="unknown")
    except ValueError as exc:
        assert "Unsupported candidate fingerprint mode" in str(exc)
    else:
        raise AssertionError("expected unsupported fingerprint mode to raise")
