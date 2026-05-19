"""Tests for the Allegro backend runtime skeleton."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms

from hotspot_al.lammps.allegro_lammps import AllegroBackend
from hotspot_al.training.allegro_runner import (
    AllegroRunner,
    build_allegro_export_command,
    build_allegro_train_command,
)


def test_allegro_runner_builds_train_and_export_commands_from_templates() -> None:
    config = {
        "allegro": {
            "train_config_path": "configs/allegro.yaml",
            "train_command_template": "python train.py --config {train_config_path} --dataset {dataset_dir} --output {output_dir}",
            "export_command_template": "python export.py --checkpoint {checkpoint_path} --output {output_dir}",
        }
    }

    train_command = build_allegro_train_command(Path("dataset"), Path("runs/model-a"), config=config)
    export_command = build_allegro_export_command(Path("ckpt.pth"), Path("deploy"), config=config)

    assert train_command == [
        "python",
        "train.py",
        "--config",
        "configs/allegro.yaml",
        "--dataset",
        "dataset",
        "--output",
        "runs/model-a",
    ]
    assert export_command == [
        "python",
        "export.py",
        "--checkpoint",
        "ckpt.pth",
        "--output",
        "deploy",
    ]


def test_allegro_backend_delegates_force_and_committee_evaluation() -> None:
    atoms = Atoms(symbols=["O", "H"], positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0]])

    def fake_force_evaluator(atoms: Atoms, model_path: str | Path | None, config: dict) -> np.ndarray:
        base = 1.0 if str(model_path).endswith("000.pth") else 2.0
        return np.full((len(atoms), 3), base, dtype=float)

    backend = AllegroBackend(
        config={
            "allegro": {
                "model_paths": ["allegro_000.pth", "allegro_001.pth"],
            }
        },
        runner=AllegroRunner(force_evaluator=fake_force_evaluator),
    )

    forces = backend.evaluate_forces(atoms)
    committee = backend.evaluate_committee(atoms)

    assert np.allclose(forces, np.ones((2, 3)))
    assert committee.shape == (2, 2, 3)
    assert np.allclose(committee[1], np.full((2, 3), 2.0))


def test_allegro_backend_train_and_export_return_dry_run_commands(tmp_path: Path) -> None:
    config = {
        "allegro": {
            "dataset_dir": tmp_path / "dataset",
            "train_output_dir": tmp_path / "runs/model-a",
            "checkpoint_path": tmp_path / "runs/model-a/best.pth",
            "train_command_template": "python train.py --dataset {dataset_dir} --output {output_dir}",
            "export_command_template": "python export.py --checkpoint {checkpoint_path} --output {output_dir}",
        }
    }
    backend = AllegroBackend(config=config)

    train_command = backend.train()
    export_command = backend.export_model(tmp_path / "deploy")

    assert train_command == [
        "python",
        "train.py",
        "--dataset",
        str(tmp_path / "dataset"),
        "--output",
        str(tmp_path / "runs/model-a"),
    ]
    assert export_command == [
        "python",
        "export.py",
        "--checkpoint",
        str(tmp_path / "runs/model-a/best.pth"),
        "--output",
        str(tmp_path / "deploy"),
    ]
