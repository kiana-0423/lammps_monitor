"""Thin runtime hooks for Allegro evaluation, training, and export."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Any, Callable

import numpy as np
from ase import Atoms

ForceEvaluator = Callable[[Atoms, str | Path | None, dict[str, Any]], np.ndarray]


def _build_command_from_template(
    template: str | None,
    *,
    context: dict[str, str],
    config_key: str,
) -> list[str]:
    """Render a shell command template into argv form."""

    if not template:
        msg = f"Missing allegro.{config_key}; provide a command template for the external Allegro runtime."
        raise ValueError(msg)
    try:
        command = template.format(**context)
    except KeyError as exc:
        missing = exc.args[0]
        msg = f"allegro.{config_key} references unknown placeholder {missing!r}."
        raise ValueError(msg) from exc
    return shlex.split(command)


def build_allegro_train_command(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    config: dict[str, Any],
) -> list[str]:
    """Build an external Allegro training command from a user template."""

    allegro_cfg = config.get("allegro", {})
    return _build_command_from_template(
        allegro_cfg.get("train_command_template"),
        context={
            "dataset_dir": str(dataset_dir),
            "output_dir": str(output_dir),
            "train_config_path": str(allegro_cfg.get("train_config_path") or ""),
        },
        config_key="train_command_template",
    )


def run_allegro_training(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    config: dict[str, Any],
    dry_run: bool = True,
) -> list[str] | subprocess.CompletedProcess[str]:
    """Run Allegro training or return the command in dry-run mode."""

    command = build_allegro_train_command(dataset_dir, output_dir, config=config)
    if dry_run:
        return command
    return subprocess.run(command, check=True, text=True, capture_output=True)


def build_allegro_export_command(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    config: dict[str, Any],
) -> list[str]:
    """Build an external Allegro export command from a user template."""

    allegro_cfg = config.get("allegro", {})
    return _build_command_from_template(
        allegro_cfg.get("export_command_template"),
        context={
            "checkpoint_path": str(checkpoint_path),
            "output_dir": str(output_dir),
        },
        config_key="export_command_template",
    )


def run_allegro_export(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    config: dict[str, Any],
    dry_run: bool = True,
) -> list[str] | subprocess.CompletedProcess[str]:
    """Run Allegro export or return the command in dry-run mode."""

    command = build_allegro_export_command(checkpoint_path, output_dir, config=config)
    if dry_run:
        return command
    return subprocess.run(command, check=True, text=True, capture_output=True)


class AllegroRunner:
    """Thin runtime wrapper for injected Allegro integration hooks."""

    def __init__(self, *, force_evaluator: ForceEvaluator | None = None, inference: Any | None = None) -> None:
        self.force_evaluator = force_evaluator
        self.inference = inference

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        force_evaluator: ForceEvaluator | None = None,
    ) -> "AllegroRunner":
        """Create a runner from config, wiring ``AllegroInference`` by default.

        Passing ``force_evaluator`` keeps the existing dependency-injection
        path. Without it, the runner builds an ``AllegroInference`` adapter
        from ``allegro.deployed_model_paths`` or ``allegro.model_paths``.
        """

        if force_evaluator is not None:
            return cls(force_evaluator=force_evaluator)

        model_paths = _resolve_inference_model_paths(config)
        if not model_paths:
            msg = "AllegroRunner.from_config requires allegro.deployed_model_paths or allegro.model_paths."
            raise ValueError(msg)

        from hotspot_al.backends.allegro_inference import AllegroInference

        inference = AllegroInference(
            model_paths,
            device=str(config.get("allegro", {}).get("device", "auto")),
            type_map=config.get("lammps", {}).get("type_map"),
        )
        return cls(force_evaluator=inference.make_evaluator(), inference=inference)

    def evaluate_forces(
        self,
        atoms: Atoms,
        *,
        config: dict[str, Any],
        model_path: str | Path | None = None,
    ) -> np.ndarray:
        """Evaluate one model if a force callback has been injected."""

        if self.force_evaluator is None:
            msg = "No Allegro force evaluator is configured. Inject AllegroRunner(force_evaluator=...) to enable online inference."
            raise NotImplementedError(msg)
        forces = np.asarray(self.force_evaluator(atoms, model_path, config), dtype=float)
        expected_shape = (len(atoms), 3)
        if forces.shape != expected_shape:
            msg = f"Expected Allegro force evaluator output with shape {expected_shape}, got {forces.shape}."
            raise ValueError(msg)
        return forces

    def evaluate_committee(
        self,
        atoms: Atoms,
        *,
        config: dict[str, Any],
        model_paths: list[str],
    ) -> np.ndarray:
        """Evaluate multiple models and return a stacked force tensor."""

        if not model_paths:
            msg = "Committee evaluation requires at least one Allegro model path."
            raise ValueError(msg)
        predictions = [self.evaluate_forces(atoms, config=config, model_path=model_path) for model_path in model_paths]
        return np.stack(predictions, axis=0)

    def train(
        self,
        *,
        config: dict[str, Any],
        dry_run: bool = True,
    ) -> list[str] | subprocess.CompletedProcess[str]:
        """Build or launch an external Allegro training command."""

        allegro_cfg = config.get("allegro", {})
        dataset_dir = allegro_cfg.get("dataset_dir")
        output_dir = allegro_cfg.get("train_output_dir")
        if dataset_dir is None or output_dir is None:
            msg = "Allegro training requires allegro.dataset_dir and allegro.train_output_dir."
            raise ValueError(msg)
        return run_allegro_training(dataset_dir, output_dir, config=config, dry_run=dry_run)

    def export_model(
        self,
        output_dir: str | Path,
        *,
        config: dict[str, Any],
        dry_run: bool = True,
    ) -> list[str] | subprocess.CompletedProcess[str]:
        """Build or launch an external Allegro export command."""

        checkpoint_path = config.get("allegro", {}).get("checkpoint_path")
        if checkpoint_path is None:
            msg = "Allegro model export requires allegro.checkpoint_path."
            raise ValueError(msg)
        return run_allegro_export(checkpoint_path, output_dir, config=config, dry_run=dry_run)


def _resolve_inference_model_paths(config: dict[str, Any]) -> list[str | Path]:
    allegro_cfg = config.get("allegro", {})
    deployed = allegro_cfg.get("deployed_model_paths") or []
    model_paths = deployed or allegro_cfg.get("model_paths") or []
    if model_paths:
        return list(model_paths)
    checkpoint_path = allegro_cfg.get("checkpoint_path")
    return [] if checkpoint_path is None else [checkpoint_path]
