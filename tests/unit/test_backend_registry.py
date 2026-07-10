"""Contract tests for platform backend registration and construction."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

from hotspot_al.backends import (
    BackendRegistry,
    BackendRole,
    CP2KBackend,
    ExecutionRequest,
    LAMMPSBackend,
    MLIPBackend,
    RealAllegroBackend,
    RuntimeStatus,
    SlurmSchedulerBackend,
    backend_engine,
    create_backend,
)
from hotspot_al.config import load_config


class ZeroMLIPBackend(MLIPBackend):
    backend_name = "zero"
    role = BackendRole.MLIP

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ZeroMLIPBackend":
        return cls()

    def check_runtime(self) -> RuntimeStatus:
        return RuntimeStatus(self.backend_name, self.role, True, "test backend")

    def evaluate_forces(self, atoms: Atoms, *, model: str | Path | None = None) -> np.ndarray:
        return np.zeros((len(atoms), 3), dtype=float)


def test_registry_constructs_third_party_backend_without_core_changes() -> None:
    registry = BackendRegistry()
    registry.register("mlip", "zero", ZeroMLIPBackend.from_config)

    backend = registry.create("mlip", "zero", {"backend": {"mlip": {"engine": "zero"}}})

    assert isinstance(backend, ZeroMLIPBackend)
    assert backend.evaluate_forces(Atoms("H")).shape == (1, 3)


def test_builtin_factory_uses_nested_backend_selection() -> None:
    config = load_config()

    assert isinstance(create_backend(config, "md"), LAMMPSBackend)
    assert isinstance(create_backend(config, "mlip"), RealAllegroBackend)
    assert isinstance(create_backend(config, "dft"), CP2KBackend)


def test_cp2k_backend_builds_workdir_relative_execution_paths() -> None:
    backend = CP2KBackend(config={"cp2k": {"executable": "/opt/cp2k/bin/cp2k.psmp"}})

    request = backend.execution_request(Path("tasks/job.inp"), output_file=Path("tasks/job.out"))

    assert request.command == ("/opt/cp2k/bin/cp2k.psmp", "-i", "job.inp")
    assert request.work_dir == Path("tasks")
    assert request.stdout_path == Path("job.out")


def test_batch_scheduler_places_resource_directives_before_shell_commands() -> None:
    scheduler = SlurmSchedulerBackend.from_config({})
    request = ExecutionRequest.from_command(
        ["solver", "input.in"],
        work_dir="job",
        resources={"directives": "#SBATCH --time=00:10:00"},
    )

    lines = scheduler._render_script(request).splitlines()

    assert lines.index("#SBATCH --time=00:10:00") < lines.index("set -euo pipefail")


def test_backend_engine_accepts_legacy_selection_during_migration() -> None:
    assert backend_engine({"backend": {"mlip": "mace"}}, "mlip") == "mace"
    assert backend_engine({"backend": {"md_engine": "openmm"}}, "md") == "openmm"


def test_load_config_normalizes_legacy_backend_override(tmp_path: Path) -> None:
    override = tmp_path / "legacy.yaml"
    override.write_text("backend:\n  md_engine: openmm\n", encoding="utf-8")

    config = load_config(override)

    assert config["backend"]["md"] == {"engine": "openmm"}
