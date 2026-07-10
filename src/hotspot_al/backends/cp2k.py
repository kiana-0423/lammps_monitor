"""CP2K implementation of the platform DFT contract."""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from hotspot_al.backends.base import BackendRole, DFTBackend, ExecutionRequest, RuntimeStatus
from hotspot_al.cp2k.cp2k_force_parser import parse_cp2k_forces
from hotspot_al.cp2k.cp2k_input import write_cp2k_inputs
from hotspot_al.cp2k.cp2k_runner import build_cp2k_command
from hotspot_al.models import ExtractedRegion


class CP2KBackend(DFTBackend):
    """Generate, execute, and parse CP2K labeling calculations."""

    backend_name = "cp2k"
    role = BackendRole.DFT

    def __init__(self, *, config: Mapping[str, Any] | None = None, executable: str | Path | None = None) -> None:
        self.config = dict(config or {})
        configured = self.config.get("cp2k", {})
        configured_executable = configured.get("executable") if isinstance(configured, Mapping) else None
        self.executable = str(executable or configured_executable or "cp2k.popt")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "CP2KBackend":
        return cls(config=config)

    def check_runtime(self) -> RuntimeStatus:
        resolved = shutil.which(self.executable)
        return RuntimeStatus(
            backend=self.backend_name,
            role=self.role,
            available=resolved is not None,
            detail=resolved or f"configured executable not found: {self.executable}",
        )

    def prepare_inputs(
        self,
        region: ExtractedRegion,
        output_dir: Path,
        *,
        task_id: str,
    ) -> Mapping[str, Path]:
        return write_cp2k_inputs(region, output_dir, config=self.config, job_name=task_id)

    def execution_request(self, input_file: Path, *, output_file: Path) -> ExecutionRequest:
        config = {**self.config, "cp2k": {**dict(self.config.get("cp2k", {})), "executable": self.executable}}
        return ExecutionRequest.from_command(
            build_cp2k_command(input_file.name, config=config),
            work_dir=input_file.parent,
            stdout_path=output_file.name,
            stderr_path=input_file.with_suffix(".err").name,
            metadata={"engine": self.backend_name, "input_file": str(input_file)},
        )

    def parse_forces(self, output_path: str | Path) -> np.ndarray:
        return parse_cp2k_forces(output_path)

    def output_is_complete(self, output_text: str) -> bool:
        markers = ("SCF run converged", "SCF converged", "ENERGY|")
        return any(marker in output_text for marker in markers)

    def evaluate_forces(self, *_args: Any, **_kwargs: Any) -> np.ndarray:
        """Compatibility guard for the former force-backend skeleton."""

        raise NotImplementedError(
            "CP2KBackend.evaluate_forces requires an external CP2K job runner; "
            "use prepare_inputs/execution_request/parse_forces."
        )


__all__ = ["CP2KBackend"]
