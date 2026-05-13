"""Candidate pool management for CP2K labeling selection."""

from __future__ import annotations

from dataclasses import dataclass, field

from hotspot_al.active_learning.deduplication import CandidateFingerprint, deduplicate_candidates, pair_distance_histogram
from hotspot_al.models import ExtractedRegion


@dataclass(slots=True)
class CandidatePool:
    """Store extracted regions and prune them by score and diversity."""

    diversity_threshold: float = 0.1
    max_candidates: int = 50
    candidates: list[CandidateFingerprint] = field(default_factory=list)

    def add(self, region: ExtractedRegion, *, score: float, metadata: dict | None = None) -> None:
        fingerprint = pair_distance_histogram(region)
        self.candidates.append(
            CandidateFingerprint(region=region, score=score, fingerprint=fingerprint, metadata=dict(metadata or {}))
        )

    def select(self) -> list[CandidateFingerprint]:
        unique = deduplicate_candidates(self.candidates, diversity_threshold=self.diversity_threshold)
        return unique[: self.max_candidates]
