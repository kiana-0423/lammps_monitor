"""Allegro implementation of the platform MLIP contract."""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

from hotspot_al.backends.base import BackendRole, MLIPBackend, RuntimeStatus
from hotspot_al.training.allegro_runner import AllegroRunner


class RealAllegroBackend(MLIPBackend):
    """Adapter that isolates Allegro-specific configuration and runtime calls."""

    backend_name = "allegro"
    role = BackendRole.MLIP

    def __init__(self, *, runner: AllegroRunner | None = None, config: Mapping[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.runner = runner or AllegroRunner()

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "RealAllegroBackend":
        copied = dict(config)
        return cls(runner=AllegroRunner.from_config(copied), config=copied)

    def check_runtime(self) -> RuntimeStatus:
        missing = [name for name in ("torch", "nequip", "allegro") if importlib.util.find_spec(name) is None]
        return RuntimeStatus(
            backend=self.backend_name,
            role=self.role,
            available=not missing,
            detail="runtime importable" if not missing else f"missing Python modules: {', '.join(missing)}",
            metadata={"models": [str(path) for path in self.model_paths()]},
        )

    def evaluate_forces(
        self,
        atoms: Atoms,
        *,
        model: str | Path | None = None,
        config: dict[str, Any] | None = None,
        model_path: str | Path | None = None,
    ) -> np.ndarray:
        """Evaluate one model; legacy keyword arguments remain accepted."""

        forces = self.runner.evaluate_forces(
            atoms,
            config=config or self.config,
            model_path=model if model is not None else model_path,
        )
        if not np.isfinite(forces).all():
            raise ValueError("Allegro returned non-finite forces.")
        return forces

    def evaluate_committee(
        self,
        atoms: Atoms,
        *,
        models: Sequence[str | Path] | None = None,
        config: dict[str, Any] | None = None,
        model_paths: list[str] | None = None,
    ) -> np.ndarray:
        selected = list(models or model_paths or self.model_paths())
        if not selected:
            raise ValueError("Allegro committee evaluation requires at least one model.")
        predictions = self.runner.evaluate_committee(
            atoms,
            config=config or self.config,
            model_paths=[str(path) for path in selected],
        )
        if not np.isfinite(predictions).all():
            raise ValueError("Allegro committee returned non-finite forces.")
        return predictions

    def model_paths(self) -> tuple[Path, ...]:
        allegro_cfg = self.config.get("allegro", {})
        if not isinstance(allegro_cfg, Mapping):
            return ()
        selected = allegro_cfg.get("deployed_model_paths") or allegro_cfg.get("model_paths") or []
        if not selected and allegro_cfg.get("checkpoint_path") is not None:
            selected = [allegro_cfg["checkpoint_path"]]
        return tuple(Path(path) for path in selected)

    def train(self, dataset_dir: Path, output_dir: Path, *, dry_run: bool = True) -> Any:
        section = self.config.setdefault("allegro", {})
        section["dataset_dir"] = str(dataset_dir)
        section["train_output_dir"] = str(output_dir)
        return self.runner.train(config=self.config, dry_run=dry_run)

    def export_model(self, checkpoint: Path | None, output_dir: Path, *, dry_run: bool = True) -> Any:
        if checkpoint is not None:
            self.config.setdefault("allegro", {})["checkpoint_path"] = str(checkpoint)
        return self.runner.export_model(output_dir, config=self.config, dry_run=dry_run)

    def reload(self, model_paths: Sequence[str | Path]) -> None:
        selected = [str(path) for path in model_paths]
        self.config.setdefault("allegro", {})["deployed_model_paths"] = selected
        inference = getattr(self.runner, "inference", None)
        if inference is not None and hasattr(inference, "reload"):
            inference.reload(selected)


__all__ = ["RealAllegroBackend"]
