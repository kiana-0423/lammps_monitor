"""Tests for automatic retraining triggers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import write

from hotspot_al.config import load_config
from hotspot_al.training.retrain_trigger import RetrainTrigger


class _FakeRunner:
    def __init__(self) -> None:
        self.trained = False
        self.exported = False

    def train(self, *, config: dict[str, Any], dry_run: bool = True):
        self.trained = True
        return ["train", config["allegro"]["dataset_dir"], str(dry_run)]

    def export_model(self, output_dir, *, config: dict[str, Any], dry_run: bool = True):
        self.exported = True
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return ["export", str(output_dir), str(dry_run)]


def test_retrain_trigger_manual_run_merges_samples(tmp_path: Path) -> None:
    labeled_dir = tmp_path / "labels"
    labeled_dir.mkdir()
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]])
    atoms.arrays["forces"] = np.zeros((2, 3))
    write(labeled_dir / "sample.extxyz", atoms, format="extxyz")
    config = load_config()
    config["retraining"] = {"dry_run": True}
    runner = _FakeRunner()

    result = RetrainTrigger(
        config=config,
        runner=runner,  # type: ignore[arg-type]
        labeled_dir=labeled_dir,
        dataset_dir=tmp_path / "dataset",
    ).trigger_now()

    assert result.triggered
    assert result.reason == "manual"
    assert result.dataset_path == tmp_path / "dataset" / "train.extxyz"
    assert result.dataset_path.is_file()
    assert runner.trained
    assert runner.exported
