"""Automatic retraining triggers for accumulated labeled samples."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ase.io import read, write

from hotspot_al.training.allegro_runner import AllegroRunner
from hotspot_al.training.model_registry import ModelRegistry, ModelVersion


@dataclass(slots=True)
class RetrainResult:
    """Outcome of one retraining check."""

    triggered: bool
    reason: str
    sample_count: int
    dataset_path: Path | None = None
    train_result: Any | None = None
    export_result: Any | None = None
    model_version: ModelVersion | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class RetrainTrigger:
    """Collect labeled extxyz samples and trigger Allegro retraining."""

    def __init__(
        self,
        *,
        config: dict[str, Any],
        runner: AllegroRunner | None = None,
        registry: ModelRegistry | None = None,
        labeled_dir: str | Path | None = None,
        dataset_dir: str | Path | None = None,
        state_path: str | Path | None = None,
    ) -> None:
        retrain_cfg = config.get("retraining", {})
        allegro_cfg = config.get("allegro", {})
        cp2k_cfg = config.get("cp2k", {})
        self.config = config
        self.runner = runner or AllegroRunner()
        self.registry = registry
        self.labeled_dir = Path(labeled_dir or cp2k_cfg.get("labeled_dataset_dir", "./labeled_data"))
        self.dataset_dir = Path(dataset_dir or allegro_cfg.get("dataset_dir") or "./training_data")
        self.state_path = Path(state_path or retrain_cfg.get("state_path", self.dataset_dir / "retrain_state.json"))
        self.min_new_samples = int(retrain_cfg.get("min_new_samples", retrain_cfg.get("sample_trigger", 10)))
        self.interval_hours = float(retrain_cfg.get("interval_hours", 24.0))
        self.dry_run = bool(retrain_cfg.get("dry_run", True))
        self.export_dir = Path(retrain_cfg.get("export_dir", allegro_cfg.get("export_dir", self.dataset_dir / "exports")))

    def check_and_run(self, *, force: bool = False) -> RetrainResult:
        """Check trigger conditions and run training if needed."""

        samples = self.collect_labeled_samples()
        state = self._load_state()
        previous_count = int(state.get("sample_count", 0))
        new_count = max(0, len(samples) - previous_count)
        reason = self._trigger_reason(force=force, new_count=new_count, state=state)
        if reason is None:
            return RetrainResult(False, "not_due", len(samples), metadata={"new_samples": new_count})
        if not samples:
            return RetrainResult(False, "no_samples", 0)

        dataset_path = self.merge_samples(samples)
        self.config.setdefault("allegro", {})["dataset_dir"] = str(self.dataset_dir)
        self.config["allegro"].setdefault("train_output_dir", str(self.dataset_dir / "runs"))
        train_result = self.runner.train(config=self.config, dry_run=self.dry_run)
        export_result = self.runner.export_model(self.export_dir, config=self.config, dry_run=self.dry_run)
        model_version = self._register_exported_model(sample_count=len(samples))
        self._write_state({"sample_count": len(samples), "last_run_at": _now(), "last_reason": reason})
        return RetrainResult(
            True,
            reason,
            len(samples),
            dataset_path=dataset_path,
            train_result=train_result,
            export_result=export_result,
            model_version=model_version,
            metadata={"new_samples": new_count},
        )

    def trigger_now(self) -> RetrainResult:
        """Manually trigger retraining."""

        return self.check_and_run(force=True)

    def collect_labeled_samples(self) -> list[Path]:
        """Return labeled extxyz files in deterministic order."""

        if not self.labeled_dir.exists():
            return []
        return sorted(path for path in self.labeled_dir.rglob("*.extxyz") if path.is_file())

    def merge_samples(self, samples: list[Path]) -> Path:
        """Merge extxyz samples into the configured Allegro dataset directory."""

        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        atoms_list = []
        for sample in samples:
            loaded = read(sample, index=":", format="extxyz")
            if isinstance(loaded, list):
                atoms_list.extend(loaded)
            else:
                atoms_list.append(loaded)
        dataset_path = self.dataset_dir / "train.extxyz"
        write(dataset_path, atoms_list, format="extxyz")
        return dataset_path

    def _trigger_reason(self, *, force: bool, new_count: int, state: dict[str, Any]) -> str | None:
        if force:
            return "manual"
        if new_count >= self.min_new_samples:
            return "sample_count"
        last_run_at = state.get("last_run_at")
        if last_run_at is None and new_count > 0:
            return "initial_samples"
        if last_run_at is not None and new_count > 0:
            last_run = datetime.fromisoformat(last_run_at)
            if datetime.now(timezone.utc) - last_run >= timedelta(hours=self.interval_hours):
                return "time_interval"
        return None

    def _register_exported_model(self, *, sample_count: int) -> ModelVersion | None:
        if self.registry is None or self.dry_run:
            return None
        model_path = self.config.get("retraining", {}).get("exported_model_path")
        if model_path is None:
            candidates = sorted(self.export_dir.glob("*.pth"))
            model_path = candidates[-1] if candidates else None
        if model_path is None:
            return None
        model = self.registry.register_model(model_path, training_set_size=sample_count)
        self.registry.deploy(config=self.config)
        return model

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {}
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def _write_state(self, state: dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
