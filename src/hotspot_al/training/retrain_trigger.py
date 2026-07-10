"""Automatic retraining triggers for accumulated labeled samples."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ase.io import read, write

from hotspot_al.backends.base import MLIPBackend
from hotspot_al.backends.factory import create_mlip_backend
from hotspot_al.training.model_registry import ModelRegistry, ModelVersion
from hotspot_al.utils.logging import configure_logging


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
    """Collect labeled samples and trigger the configured MLIP backend."""

    def __init__(
        self,
        *,
        config: dict[str, Any],
        mlip_backend: MLIPBackend | None = None,
        runner: Any | None = None,
        registry: ModelRegistry | None = None,
        labeled_dir: str | Path | None = None,
        dataset_dir: str | Path | None = None,
        state_path: str | Path | None = None,
    ) -> None:
        retrain_cfg = config.get("retraining", {})
        datasets_cfg = config.get("datasets", {})
        legacy_mlip_cfg = config.get("allegro", {})
        legacy_dft_cfg = config.get("cp2k", {})
        self.config = config
        if mlip_backend is not None and runner is not None:
            raise ValueError("Pass mlip_backend or the legacy runner argument, not both.")
        self.mlip_backend = mlip_backend or create_mlip_backend(config, legacy_runner=runner)
        self.runner = runner
        self.registry = registry
        self.labeled_dir = Path(
            labeled_dir
            or datasets_cfg.get("labeled_dir")
            or legacy_dft_cfg.get("labeled_dataset_dir", "./labeled_data")
        )
        self.dataset_dir = Path(
            dataset_dir
            or datasets_cfg.get("training_dir")
            or legacy_mlip_cfg.get("dataset_dir")
            or "./training_data"
        )
        self.state_path = Path(state_path or retrain_cfg.get("state_path", self.dataset_dir / "retrain_state.json"))
        self.min_new_samples = int(retrain_cfg.get("min_new_samples", retrain_cfg.get("sample_trigger", 10)))
        self.interval_hours = float(retrain_cfg.get("interval_hours", 24.0))
        self.dry_run = bool(retrain_cfg.get("dry_run", True))
        self.export_dir = Path(retrain_cfg.get("export_dir", legacy_mlip_cfg.get("export_dir", self.dataset_dir / "exports")))
        self.logger = configure_logging(config, name=__name__)

    def check_and_run(self, *, force: bool = False) -> RetrainResult:
        """Check trigger conditions and run training if needed."""

        samples = self.collect_labeled_samples()
        state = self._load_state()
        previous_count = int(state.get("sample_count", 0))
        new_count = max(0, len(samples) - previous_count)
        reason = self._evaluate_trigger_condition(force=force, new_count=new_count, state=state)
        if reason is None:
            self.logger.info("retraining not due: samples=%d new_samples=%d", len(samples), new_count)
            return RetrainResult(False, "not_due", len(samples), metadata={"new_samples": new_count})
        if not samples:
            self.logger.info("retraining skipped: no labeled samples found in %s", self.labeled_dir)
            return RetrainResult(False, "no_samples", 0)

        self.logger.info("retraining triggered reason=%s samples=%d new_samples=%d", reason, len(samples), new_count)
        dataset_path = self.merge_samples(samples)
        train_result, export_result = self._execute_training()
        model_version = self._register_exported_model(sample_count=len(samples))
        self._update_state(sample_count=len(samples), reason=reason)
        self.logger.info("retraining finished reason=%s dataset=%s", reason, dataset_path)
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
        """Merge extxyz samples into the configured platform dataset directory."""

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

    def _evaluate_trigger_condition(self, *, force: bool, new_count: int, state: dict[str, Any]) -> str | None:
        """Return a retraining reason when current state is due."""

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

    def _execute_training(self) -> tuple[Any, Any]:
        """Run training and export through the configured MLIP contract."""

        output_dir = Path(self.config.get("retraining", {}).get("train_output_dir", self.dataset_dir / "runs"))
        checkpoint_value = self.config.get("retraining", {}).get("checkpoint_path")
        if checkpoint_value is None:
            checkpoint_value = self.config.get("allegro", {}).get("checkpoint_path")
        checkpoint = None if checkpoint_value is None else Path(checkpoint_value)
        train_result = self.mlip_backend.train(self.dataset_dir, output_dir, dry_run=self.dry_run)
        export_result = self.mlip_backend.export_model(checkpoint, self.export_dir, dry_run=self.dry_run)
        return train_result, export_result

    def _update_state(self, *, sample_count: int, reason: str) -> None:
        """Persist retraining bookkeeping."""

        self._write_state({"sample_count": sample_count, "last_run_at": _now(), "last_reason": reason})

    def _register_exported_model(self, *, sample_count: int) -> ModelVersion | None:
        if self.registry is None or self.dry_run:
            return None
        model_path = self.config.get("retraining", {}).get("exported_model_path")
        if model_path is None:
            candidates = sorted(self.export_dir.glob("*.pth"), key=lambda path: (path.stat().st_mtime, path.name))
            model_path = candidates[-1] if candidates else None
        if model_path is None:
            return None
        model = self.registry.register_model(model_path, training_set_size=sample_count)
        self.registry.deploy(inference=self.mlip_backend)
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
