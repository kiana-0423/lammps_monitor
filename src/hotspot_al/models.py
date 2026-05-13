"""Shared data structures used across the hotspot active learning pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from ase import Atoms


@dataclass(slots=True)
class FrameData:
    """A single trajectory frame with optional dynamics metadata."""

    atoms: Atoms
    step: int
    time: float | None = None
    forces: np.ndarray | None = None
    velocities: np.ndarray | None = None
    energy: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OODFrameResult:
    """Atom-wise out-of-distribution scoring output for one frame."""

    atom_scores: np.ndarray
    metric_scores: dict[str, np.ndarray]
    max_score: float
    hotspot_indices: list[int]
    trigger_reason: list[str]
    triggered: bool
    stage: str = "full"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Hotspot:
    """A merged cluster of anomalous atoms."""

    core_atom_indices: list[int]
    center: np.ndarray
    max_score: float
    trigger_reasons: list[str]
    step: int
    event_id: str | None = None
    backend: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def atom_indices(self) -> list[int]:
        """Alias used by the PHAL/HAL workflow description."""

        return self.core_atom_indices


@dataclass(slots=True)
class ExtractedRegion:
    """A local cluster or slab extracted around a hotspot."""

    atoms: Atoms
    original_indices: list[int]
    core_indices: list[int]
    inner_buffer_indices: list[int]
    outer_buffer_indices: list[int]
    boundary_indices: list[int]
    h_cap_indices: list[int]
    hotspot_indices: list[int]
    region_labels: list[str] = field(default_factory=list)
    mask_weights: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EventRecord:
    """Buffered frames collected around an OOD trigger event."""

    pre_frames: list[FrameData]
    trigger_frame: FrameData
    post_frames: list[FrameData]
    hotspot_atoms: list[int]
    ood_scores: np.ndarray
    trigger_reason: list[str]
    step: int
    time: float | None
    event_id: str | None = None
    backend: str | None = None
    model_version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
