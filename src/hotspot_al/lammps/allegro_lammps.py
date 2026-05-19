"""Allegro backend hooks for LAMMPS active learning workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase import Atoms

from hotspot_al.lammps.backend import MLIPBackend
from hotspot_al.lammps.lammps_input import build_lammps_input
from hotspot_al.training.allegro_adapter import write_allegro_dataset
from hotspot_al.training.allegro_runner import AllegroRunner


class AllegroBackend(MLIPBackend):
    """Allegro backend adapter."""

    name = "allegro"

    def __init__(self, *, config: dict[str, Any] | None = None, runner: AllegroRunner | None = None) -> None:
        self.config = config or {}
        self.runner = runner or AllegroRunner()

    def _resolve_config(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        if config is not None:
            return config
        if self.config:
            return self.config
        msg = "Allegro backend requires a config argument or an initialized config."
        raise ValueError(msg)

    def _resolve_model_paths(self, config: dict[str, Any]) -> list[str]:
        allegro_cfg = config.get("allegro", {})
        model_paths = allegro_cfg.get("deployed_model_paths") or allegro_cfg.get("model_paths", [])
        if not model_paths:
            msg = "Allegro backend requires at least one model path."
            raise ValueError(msg)
        return [str(path) for path in model_paths]

    def write_lammps_input(self, config: dict[str, Any] | None = None) -> str:
        resolved_config = self._resolve_config(config)
        allegro_cfg = resolved_config.get("allegro", {})
        model_paths = self._resolve_model_paths(resolved_config)
        pair_style = f"pair_style {allegro_cfg.get('lammps_pair_style', 'allegro')} {model_paths[0]}"
        pair_coeff = "pair_coeff * *"
        return build_lammps_input("\n".join([pair_style, pair_coeff]), config=resolved_config)

    def evaluate_forces(self, atoms: Atoms):
        config = self._resolve_config()
        model_paths = self._resolve_model_paths(config)
        return self.runner.evaluate_forces(atoms, config=config, model_path=model_paths[0])

    def evaluate_committee(self, atoms: Atoms, model_paths: list[str] | None = None):
        config = self._resolve_config()
        resolved_paths = [str(path) for path in model_paths] if model_paths is not None else self._resolve_model_paths(config)
        return self.runner.evaluate_committee(atoms, config=config, model_paths=resolved_paths)

    def write_training_data(self, region, *, forces, output_dir, config, filename: str = "dataset.extxyz"):
        return write_allegro_dataset(region, forces=forces, output_dir=output_dir, config=config, filename=filename)

    def train(self, config: dict[str, Any] | None = None):
        return self.runner.train(config=self._resolve_config(config))

    def export_model(self, output_dir: str | Path):
        return self.runner.export_model(output_dir, config=self._resolve_config())
