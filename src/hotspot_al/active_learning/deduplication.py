"""Candidate deduplication using lightweight geometric fingerprints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from ase.data import atomic_numbers
from scipy.spatial.distance import pdist

from hotspot_al.models import ExtractedRegion


def pair_distance_histogram(region: ExtractedRegion, *, bins: int = 16, cutoff: float = 8.0) -> np.ndarray:
    """Compute a simple pair-distance histogram fingerprint."""

    positions = region.atoms.get_positions()
    if len(positions) < 2:
        return np.zeros(bins, dtype=float)
    distances = pdist(positions)
    histogram, _ = np.histogram(distances, bins=bins, range=(0.0, cutoff), density=True)
    return histogram.astype(float)


def type_weighted_pair_distance_histogram(
    region: ExtractedRegion,
    *,
    bins: int = 16,
    cutoff: float = 8.0,
) -> np.ndarray:
    """Compute a pair-distance histogram weighted by atom-type pairs."""

    positions = region.atoms.get_positions()
    if len(positions) < 2:
        return np.zeros(bins * 3, dtype=float)
    symbols = region.atoms.get_chemical_symbols()
    distances = pdist(positions)
    numbers = np.fromiter((atomic_numbers.get(symbol, 0) for symbol in symbols), dtype=float, count=len(symbols))
    left, right = np.triu_indices(len(symbols), k=1)
    weights = 0.5 * (numbers[left] + numbers[right])
    plain, _ = np.histogram(distances, bins=bins, range=(0.0, cutoff), density=True)
    weighted_sum, _ = np.histogram(distances, bins=bins, range=(0.0, cutoff), weights=weights)
    counts, _ = np.histogram(distances, bins=bins, range=(0.0, cutoff))
    weighted_mean = np.divide(weighted_sum, counts, out=np.zeros_like(weighted_sum, dtype=float), where=counts > 0)
    return np.concatenate([plain.astype(float), weighted_sum.astype(float), weighted_mean.astype(float)])


def fingerprint_region(region: ExtractedRegion, *, mode: str = "pair_distance_histogram") -> np.ndarray:
    """Compute a candidate fingerprint by configured mode."""

    normalized = mode.lower()
    if normalized in {"pair_distance_histogram", "pair_distance", "distance"}:
        return pair_distance_histogram(region)
    if normalized in {"type_weighted_pair_distance_histogram", "type_weighted_pair_distance", "weighted_pair"}:
        return type_weighted_pair_distance_histogram(region)
    msg = f"Unsupported candidate fingerprint mode: {mode}"
    raise ValueError(msg)


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
