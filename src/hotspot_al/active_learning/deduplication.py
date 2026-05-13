"""Candidate deduplication using lightweight geometric fingerprints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hotspot_al.models import ExtractedRegion


def pair_distance_histogram(region: ExtractedRegion, *, bins: int = 16, cutoff: float = 8.0) -> np.ndarray:
    """Compute a simple pair-distance histogram fingerprint."""

    positions = region.atoms.get_positions()
    distances: list[float] = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            distances.append(float(np.linalg.norm(positions[i] - positions[j])))
    if not distances:
        return np.zeros(bins, dtype=float)
    histogram, _ = np.histogram(distances, bins=bins, range=(0.0, cutoff), density=True)
    return histogram.astype(float)


def fingerprint_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Return an L2 distance between fingerprints."""

    return float(np.linalg.norm(np.asarray(left, dtype=float) - np.asarray(right, dtype=float)))


@dataclass(slots=True)
class CandidateFingerprint:
    """Container storing one candidate region and its fingerprint."""

    region: ExtractedRegion
    score: float
    fingerprint: np.ndarray
    metadata: dict[str, Any]


def deduplicate_candidates(
    candidates: list[CandidateFingerprint],
    *,
    diversity_threshold: float,
) -> list[CandidateFingerprint]:
    """Keep high-score diverse candidates under a distance threshold."""

    unique: list[CandidateFingerprint] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if all(fingerprint_distance(candidate.fingerprint, kept.fingerprint) >= diversity_threshold for kept in unique):
            unique.append(candidate)
    return unique
