"""Stable backend contracts for the PHAL platform.

Core algorithms depend on these interfaces, never on a concrete scientific
program.  Third-party packages can implement the contracts and register a
factory without changing PHAL's workflow modules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from ase import Atoms

from hotspot_al.models import ExtractedRegion, FrameData


class BackendRole(str, Enum):
    """Roles that can be filled by built-in or third-party backends."""

    MD = "md"
    MLIP = "mlip"
    DFT = "dft"
    SCHEDULER = "scheduler"


class JobState(str, Enum):
    """Portable state vocabulary used by scheduler adapters."""

    PREPARED = "prepared"
    SUBMITTED = "submitted"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class RuntimeStatus:
    """Result of a backend runtime health check."""

    backend: str
    role: BackendRole
    available: bool
    detail: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ExecutionRequest:
    """Scheduler-neutral request to execute one external command."""

    command: tuple[str, ...]
    work_dir: Path
    stdout_path: Path | None = None
    stderr_path: Path | None = None
    environment: Mapping[str, str] = field(default_factory=dict)
    resources: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_command(
        cls,
        command: Sequence[str | Path],
        *,
        work_dir: str | Path,
        stdout_path: str | Path | None = None,
        stderr_path: str | Path | None = None,
        environment: Mapping[str, str] | None = None,
        resources: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ExecutionRequest":
        return cls(
            command=tuple(str(part) for part in command),
            work_dir=Path(work_dir),
            stdout_path=None if stdout_path is None else Path(stdout_path),
            stderr_path=None if stderr_path is None else Path(stderr_path),
            environment=dict(environment or {}),
            resources=dict(resources or {}),
            metadata=dict(metadata or {}),
        )


@dataclass(slots=True)
class BackendJob:
    """Opaque job handle shared between workflows and scheduler backends."""

    backend: str
    state: JobState
    external_id: str | None = None
    native_handle: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Backend(ABC):
    """Base class implemented by every PHAL backend."""

    role: BackendRole
    backend_name: str

    @classmethod
    @abstractmethod
    def from_config(cls, config: Mapping[str, Any]) -> "Backend":
        """Construct the backend from the complete PHAL configuration."""

    @abstractmethod
    def check_runtime(self) -> RuntimeStatus:
        """Report whether the configured runtime is usable."""


class MLIPBackend(Backend):
    """Force inference, committee evaluation, training, and export contract."""

    role = BackendRole.MLIP

    @abstractmethod
    def evaluate_forces(self, atoms: Atoms, *, model: str | Path | None = None) -> np.ndarray:
        """Return finite forces with shape ``(n_atoms, 3)``."""

    def evaluate_committee(
        self,
        atoms: Atoms,
        *,
        models: Sequence[str | Path] | None = None,
    ) -> np.ndarray:
        selected = tuple(models or self.model_paths())
        if not selected:
            raise ValueError(f"{self.backend_name} committee evaluation requires at least one model.")
        return np.stack([self.evaluate_forces(atoms, model=model) for model in selected], axis=0)

    def model_paths(self) -> tuple[Path, ...]:
        """Return configured deployed or committee model artifacts."""

        return ()

    def model_version(self) -> str | None:
        paths = self.model_paths()
        return paths[0].name if paths else None

    def train(self, dataset_dir: Path, output_dir: Path, *, dry_run: bool = True) -> Any:
        raise NotImplementedError(f"{self.backend_name} does not implement training.")

    def export_model(self, checkpoint: Path | None, output_dir: Path, *, dry_run: bool = True) -> Any:
        raise NotImplementedError(f"{self.backend_name} does not implement model export.")

    def reload(self, model_paths: Sequence[str | Path]) -> None:
        """Activate newly deployed model artifacts when hot reload is supported."""

        return None


class DFTBackend(Backend):
    """DFT input generation, command construction, and result parsing contract."""

    role = BackendRole.DFT

    @abstractmethod
    def prepare_inputs(
        self,
        region: ExtractedRegion,
        output_dir: Path,
        *,
        task_id: str,
    ) -> Mapping[str, Path]:
        """Generate all files needed for one labeling task."""

    @abstractmethod
    def execution_request(self, input_file: Path, *, output_file: Path) -> ExecutionRequest:
        """Build a scheduler-neutral request for one prepared calculation."""

    @abstractmethod
    def parse_forces(self, output_path: Path) -> np.ndarray:
        """Parse forces from a completed calculation."""

    @abstractmethod
    def output_is_complete(self, output_text: str) -> bool:
        """Return whether an output contains a successful completion marker."""


class MDBackend(Backend):
    """Molecular-dynamics input, execution, and trajectory contract."""

    role = BackendRole.MD

    @abstractmethod
    def execution_request(self, input_file: Path, *, work_dir: Path) -> ExecutionRequest:
        """Build a scheduler-neutral MD execution request."""

    @abstractmethod
    def read_frames(self, trajectory: Path) -> list[FrameData]:
        """Read frames produced by the MD engine."""


class SchedulerBackend(Backend):
    """External execution contract for Local, Slurm, PBS, or other runtimes."""

    role = BackendRole.SCHEDULER

    @abstractmethod
    def submit(self, request: ExecutionRequest) -> BackendJob:
        """Submit an execution request and return a portable handle."""

    @abstractmethod
    def poll(self, job: BackendJob) -> JobState:
        """Refresh and return the job state."""

    @abstractmethod
    def cancel(self, job: BackendJob) -> None:
        """Cancel a submitted job."""


# Compatibility alias retained for existing integrations.
ForceBackend = MLIPBackend


__all__ = [
    "Backend",
    "BackendJob",
    "BackendRole",
    "DFTBackend",
    "ExecutionRequest",
    "ForceBackend",
    "JobState",
    "MDBackend",
    "MLIPBackend",
    "RuntimeStatus",
    "SchedulerBackend",
]
