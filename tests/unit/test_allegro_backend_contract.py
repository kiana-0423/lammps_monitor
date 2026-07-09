"""Tests for the Allegro backend runtime skeleton."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from hotspot_al.backends.allegro_inference import AllegroInference
from hotspot_al.lammps.allegro_lammps import AllegroBackend
from hotspot_al.training.allegro_runner import (
    AllegroRunner,
    build_allegro_export_command,
    build_allegro_train_command,
)


def test_evaluate_forces_without_force_evaluator_raises_not_implemented() -> None:
    atoms = Atoms(symbols=["O", "H"], positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0]])
    runner = AllegroRunner()

    with pytest.raises(NotImplementedError, match="No Allegro force evaluator"):
        runner.evaluate_forces(atoms, config={})


def test_evaluate_forces_rejects_wrong_shape() -> None:
    atoms = Atoms(symbols=["O", "H"], positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0]])

    def wrong_shape(atoms: Atoms, model_path: str | Path | None, config: dict) -> np.ndarray:
        return np.zeros((len(atoms), 2))

    runner = AllegroRunner(force_evaluator=wrong_shape)
    with pytest.raises(ValueError, match="Expected Allegro force evaluator output"):
        runner.evaluate_forces(atoms, config={})


def test_evaluate_committee_rejects_empty_model_paths() -> None:
    atoms = Atoms(symbols=["O"], positions=[[0.0, 0.0, 0.0]])
    runner = AllegroRunner(force_evaluator=lambda atoms, model_path, config: np.zeros((len(atoms), 3)))

    with pytest.raises(ValueError, match="at least one Allegro model path"):
        runner.evaluate_committee(atoms, config={}, model_paths=[])


def test_evaluate_forces_returns_expected_shape() -> None:
    atoms = Atoms(symbols=["O", "H", "H"], positions=np.zeros((3, 3)))
    runner = AllegroRunner(force_evaluator=lambda atoms, model_path, config: np.ones((len(atoms), 3)))

    forces = runner.evaluate_forces(atoms, config={}, model_path="model.pth")

    assert forces.shape == (3, 3)


def test_evaluate_committee_returns_expected_shape() -> None:
    atoms = Atoms(symbols=["O", "H"], positions=np.zeros((2, 3)))

    def fake_force_evaluator(atoms: Atoms, model_path: str | Path | None, config: dict) -> np.ndarray:
        value = 1.0 if str(model_path).endswith("a.pth") else 2.0
        return np.full((len(atoms), 3), value)

    runner = AllegroRunner(force_evaluator=fake_force_evaluator)

    committee = runner.evaluate_committee(atoms, config={}, model_paths=["a.pth", "b.pth"])

    assert committee.shape == (2, 2, 3)
    assert np.allclose(committee[0], 1.0)
    assert np.allclose(committee[1], 2.0)


def test_allegro_runner_from_config_wires_inference(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    atoms = Atoms(symbols=["O", "H"], positions=np.zeros((2, 3)))
    deployed = tmp_path / "deployed.pth"
    deployed.write_bytes(b"placeholder")

    monkeypatch.setattr(AllegroInference, "_load_model", lambda self, model_path: object())
    monkeypatch.setattr(
        AllegroInference,
        "_call_model",
        lambda self, model, atoms, config: np.full((len(atoms), 3), 3.0),
    )

    runner = AllegroRunner.from_config({"allegro": {"deployed_model_paths": [str(deployed)], "device": "cpu"}})
    forces = runner.evaluate_forces(atoms, config={})

    assert runner.inference is not None
    assert np.allclose(forces, 3.0)


def test_allegro_runner_from_config_requires_model_paths() -> None:
    with pytest.raises(ValueError, match="deployed_model_paths or allegro.model_paths"):
        AllegroRunner.from_config({"allegro": {"model_paths": []}})


def test_allegro_inference_missing_model_path_is_fatal(tmp_path: Path) -> None:
    inference = AllegroInference([tmp_path / "missing.pth"], device="cpu")

    with pytest.raises(FileNotFoundError, match="does not exist"):
        inference.predict_forces(Atoms("H", positions=[[0.0, 0.0, 0.0]]))


def test_allegro_inference_runtime_failure_returns_nan_and_logs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    deployed = tmp_path / "deployed.pth"
    deployed.write_bytes(b"placeholder")
    inference = AllegroInference([deployed], device="cpu")
    atoms = Atoms("H2", positions=np.zeros((2, 3)))

    monkeypatch.setattr(inference, "_load_model", lambda model_path: object())

    def fail_call(*_args: object, **_kwargs: object) -> np.ndarray:
        raise ValueError("bad output")

    monkeypatch.setattr(inference, "_call_model", fail_call)

    forces = inference.predict_forces(atoms)

    assert np.isnan(forces).all()
    assert "returning NaN forces" in caplog.text


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


def test_unknown_command_template_placeholder_has_clear_error() -> None:
    config = {"allegro": {"train_command_template": "python train.py --bad {missing}"}}

    with pytest.raises(ValueError, match="unknown placeholder 'missing'"):
        build_allegro_train_command(Path("dataset"), Path("output"), config=config)


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


def test_allegro_backend_from_config_uses_inference(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    atoms = Atoms(symbols=["H"], positions=[[0.0, 0.0, 0.0]])
    deployed = tmp_path / "deployed.pth"
    deployed.write_bytes(b"placeholder")
    monkeypatch.setattr(AllegroInference, "_load_model", lambda self, model_path: object())
    monkeypatch.setattr(
        AllegroInference,
        "_call_model",
        lambda self, model, atoms, config: np.full((len(atoms), 3), 2.0),
    )

    backend = AllegroBackend.from_config({"allegro": {"deployed_model_paths": [str(deployed)], "device": "cpu"}})

    assert np.allclose(backend.evaluate_forces(atoms), 2.0)


def test_allegro_backend_train_dry_run_command(tmp_path: Path) -> None:
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

    assert train_command == [
        "python",
        "train.py",
        "--dataset",
        str(tmp_path / "dataset"),
        "--output",
        str(tmp_path / "runs/model-a"),
    ]


def test_allegro_backend_export_dry_run_command(tmp_path: Path) -> None:
    config = {
        "allegro": {
            "checkpoint_path": tmp_path / "runs/model-a/best.pth",
            "export_command_template": "python export.py --checkpoint {checkpoint_path} --output {output_dir}",
        }
    }
    backend = AllegroBackend(config=config)

    export_command = backend.export_model(tmp_path / "deploy")

    assert export_command == [
        "python",
        "export.py",
        "--checkpoint",
        str(tmp_path / "runs/model-a/best.pth"),
        "--output",
        str(tmp_path / "deploy"),
    ]
