"""LAMMPS implementation of the platform MD contract."""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

from hotspot_al.backends.base import BackendRole, ExecutionRequest, MDBackend, RuntimeStatus
from hotspot_al.io.lammps_reader import read_dump
from hotspot_al.lammps.lammps_runner import build_lammps_command
from hotspot_al.models import FrameData


class LAMMPSBackend(MDBackend):
    """Generate execution requests and read trajectories for LAMMPS."""

    backend_name = "lammps"
    role = BackendRole.MD

    def __init__(self, *, config: Mapping[str, Any] | None = None, executable: str | Path | None = None) -> None:
        self.config = dict(config or {})
        section = self.config.get("lammps", {})
        configured = section.get("executable") if isinstance(section, Mapping) else None
        self.executable = str(executable or configured or "lmp")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "LAMMPSBackend":
        return cls(config=config)

    def check_runtime(self) -> RuntimeStatus:
        resolved = shutil.which(self.executable)
        return RuntimeStatus(
            backend=self.backend_name,
            role=self.role,
            available=resolved is not None,
            detail=resolved or f"configured executable not found: {self.executable}",
        )

    def execution_request(self, input_file: Path, *, work_dir: Path) -> ExecutionRequest:
        config = {**self.config, "lammps": {**dict(self.config.get("lammps", {})), "executable": self.executable}}
        command_input = input_file.name if input_file.parent == work_dir else input_file
        return ExecutionRequest.from_command(
            build_lammps_command(command_input, config=config),
            work_dir=work_dir,
            metadata={"engine": self.backend_name, "input_file": str(input_file)},
        )

    def read_frames(self, trajectory: Path) -> list[FrameData]:
        section = self.config.get("lammps", {})
        type_map = section.get("type_map") if isinstance(section, Mapping) else None
        timestep_fs = section.get("timestep_fs") if isinstance(section, Mapping) else None
        return read_dump(trajectory, type_map=type_map, timestep_fs=timestep_fs)

    def evaluate_forces(
        self,
        atoms: Atoms,
        *,
        config: dict[str, Any] | None = None,
        model_path: str | Path | None = None,
    ) -> np.ndarray:
        raise NotImplementedError("LAMMPSBackend.evaluate_forces requires an external LAMMPS input/runtime adapter.")

    def read_dump_forces(self, dump_path: str | Path, *, type_map: dict[int, str] | None = None) -> np.ndarray:
        frames = read_dump(dump_path, type_map=type_map)
        if not frames or frames[0].forces is None:
            raise ValueError(f"No force-bearing LAMMPS frame found in {dump_path}.")
        return frames[0].forces


__all__ = ["LAMMPSBackend"]
