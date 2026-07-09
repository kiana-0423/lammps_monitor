"""Shared data structures used across the hotspot active learning pipeline."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from ase import Atoms


def _array_to_state(values: np.ndarray | None) -> list[Any] | None:
    if values is None:
        return None
    return np.asarray(values).tolist()


def _array_from_state(values: list[Any] | None) -> np.ndarray | None:
    if values is None:
        return None
    return np.asarray(values)


def _atoms_to_dict(atoms: Atoms) -> dict[str, Any]:
    """Convert ``Atoms`` into a JSON-like state dictionary."""

    arrays = {
        key: np.asarray(value).tolist()
        for key, value in atoms.arrays.items()
        if key not in {"numbers", "positions"}
    }
    return {
        "symbols": atoms.get_chemical_symbols(),
        "positions": atoms.get_positions().tolist(),
        "cell": atoms.cell.array.tolist(),
        "pbc": np.asarray(atoms.pbc, dtype=bool).tolist(),
        "arrays": arrays,
        "info": dict(atoms.info),
    }


def _atoms_from_dict(data: dict[str, Any]) -> Atoms:
    """Reconstruct ``Atoms`` from ``_atoms_to_dict`` output."""

    atoms = Atoms(
        symbols=list(data["symbols"]),
        positions=np.asarray(data["positions"], dtype=float),
        cell=np.asarray(data.get("cell", np.zeros((3, 3))), dtype=float),
        pbc=np.asarray(data.get("pbc", [False, False, False]), dtype=bool),
    )
    for key, value in dict(data.get("arrays", {})).items():
        atoms.arrays[key] = np.asarray(value)
    atoms.info.update(dict(data.get("info", {})))
    return atoms


def _reconstruct_frame(state: dict[str, Any]) -> "FrameData":
    frame = FrameData.__new__(FrameData)
    frame.__setstate__(state)
    return frame


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

    def __getstate__(self) -> dict[str, Any]:
        return {
            "atoms": _atoms_to_dict(self.atoms),
            "step": self.step,
            "time": self.time,
            "forces": _array_to_state(self.forces),
            "velocities": _array_to_state(self.velocities),
            "energy": self.energy,
            "metadata": self.metadata,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.atoms = _atoms_from_dict(state["atoms"])
        self.step = int(state["step"])
        self.time = state.get("time")
        self.forces = _array_from_state(state.get("forces"))
        self.velocities = _array_from_state(state.get("velocities"))
        self.energy = state.get("energy")
        self.metadata = dict(state.get("metadata", {}))


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
        """Deprecated alias for ``core_atom_indices``.

        .. deprecated::
           Use ``core_atom_indices`` instead.
        """

        warnings.warn(
            "Hotspot.atom_indices is deprecated; use core_atom_indices instead.",
            DeprecationWarning,
            stacklevel=2,
        )
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

    def __getstate__(self) -> dict[str, Any]:
        return {
            "atoms": _atoms_to_dict(self.atoms),
            "original_indices": self.original_indices,
            "core_indices": self.core_indices,
            "inner_buffer_indices": self.inner_buffer_indices,
            "outer_buffer_indices": self.outer_buffer_indices,
            "boundary_indices": self.boundary_indices,
            "h_cap_indices": self.h_cap_indices,
            "hotspot_indices": self.hotspot_indices,
            "region_labels": self.region_labels,
            "mask_weights": _array_to_state(self.mask_weights),
            "metadata": self.metadata,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.atoms = _atoms_from_dict(state["atoms"])
        self.original_indices = list(state["original_indices"])
        self.core_indices = list(state["core_indices"])
        self.inner_buffer_indices = list(state["inner_buffer_indices"])
        self.outer_buffer_indices = list(state["outer_buffer_indices"])
        self.boundary_indices = list(state["boundary_indices"])
        self.h_cap_indices = list(state["h_cap_indices"])
        self.hotspot_indices = list(state["hotspot_indices"])
        self.region_labels = list(state.get("region_labels", []))
        self.mask_weights = _array_from_state(state.get("mask_weights"))
        self.metadata = dict(state.get("metadata", {}))


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

    def __getstate__(self) -> dict[str, Any]:
        return {
            "pre_frames": [frame.__getstate__() for frame in self.pre_frames],
            "trigger_frame": self.trigger_frame.__getstate__(),
            "post_frames": [frame.__getstate__() for frame in self.post_frames],
            "hotspot_atoms": self.hotspot_atoms,
            "ood_scores": _array_to_state(self.ood_scores),
            "trigger_reason": self.trigger_reason,
            "step": self.step,
            "time": self.time,
            "event_id": self.event_id,
            "backend": self.backend,
            "model_version": self.model_version,
            "metadata": self.metadata,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.pre_frames = [_reconstruct_frame(frame_state) for frame_state in state["pre_frames"]]
        self.trigger_frame = _reconstruct_frame(state["trigger_frame"])
        self.post_frames = [_reconstruct_frame(frame_state) for frame_state in state["post_frames"]]
        self.hotspot_atoms = list(state["hotspot_atoms"])
        scores = _array_from_state(state.get("ood_scores"))
        self.ood_scores = np.asarray([] if scores is None else scores, dtype=float)
        self.trigger_reason = list(state["trigger_reason"])
        self.step = int(state["step"])
        self.time = state.get("time")
        self.event_id = state.get("event_id")
        self.backend = state.get("backend")
        self.model_version = state.get("model_version")
        self.metadata = dict(state.get("metadata", {}))
