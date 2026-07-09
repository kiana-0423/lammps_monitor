"""Candidate pool management for CP2K labeling selection."""

from __future__ import annotations

from dataclasses import dataclass, field

from hotspot_al.active_learning.deduplication import (
    CandidateFingerprint,
    deduplicate_candidates,
    fingerprint_distance,
    fingerprint_region,
)
from hotspot_al.models import ExtractedRegion


@dataclass(slots=True)
class CandidatePool:
    """Store extracted regions and prune them by score and diversity.

    When ``deduplicate`` is True, the pool maintains an incremental index of
    accepted candidates. New additions are checked against the current unique
    set immediately, so ``select()`` avoids a full O(K^2) pass on every call.
    """

    diversity_threshold: float = 0.1
    max_candidates: int = 50
    deduplicate: bool = True
    fingerprint_mode: str = "pair_distance_histogram"
    candidates: list[CandidateFingerprint] = field(default_factory=list)
    _unique: list[CandidateFingerprint] = field(default_factory=list)

    def add(self, region: ExtractedRegion, *, score: float, metadata: dict | None = None) -> None:
        fingerprint = fingerprint_region(region, mode=self.fingerprint_mode)
        entry = CandidateFingerprint(region=region, score=score, fingerprint=fingerprint, metadata=dict(metadata or {}))
        self.candidates.append(entry)
        if self.deduplicate:
            self._incremental_add(entry)

    def _incremental_add(self, entry: CandidateFingerprint) -> None:
        """Add *entry* to the unique index if it is diverse enough."""

        duplicate_indices = [
            index
            for index, kept in enumerate(self._unique)
            if fingerprint_distance(entry.fingerprint, kept.fingerprint) < self.diversity_threshold
        ]
        if not duplicate_indices:
            self._unique.append(entry)
            self._unique.sort(key=lambda item: item.score, reverse=True)
            return
        if entry.score > max(self._unique[index].score for index in duplicate_indices):
            self._unique = [kept for index, kept in enumerate(self._unique) if index not in duplicate_indices]
            self._unique.append(entry)
            self._unique.sort(key=lambda item: item.score, reverse=True)

    def _rebuild_unique(self) -> None:
        """Rebuild the unique index from scratch."""

        self._unique = deduplicate_candidates(self.candidates, diversity_threshold=self.diversity_threshold)

    def select(self) -> list[CandidateFingerprint]:
        if not self.deduplicate:
            return sorted(self.candidates, key=lambda item: item.score, reverse=True)[: self.max_candidates]
        if not self._unique and self.candidates:
            self._rebuild_unique()
        return self._unique[: self.max_candidates]
