"""Version and deploy trained Allegro models."""

from __future__ import annotations

import json
import os
import shutil
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from hotspot_al.utils.logging import configure_logging

if os.name == "nt":
    import msvcrt
else:
    import fcntl


SmokeTest = Callable[[Path], None]


@dataclass(slots=True)
class ModelVersion:
    """Metadata for one registered model version."""

    version: str
    path: Path
    created_at: str
    training_set_size: int | None = None
    validation_metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelRegistry:
    """Small filesystem-backed registry for deployed Allegro models."""

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root_dir / "registry.json"
        self.lock_path = self.root_dir / "registry.json.lock"
        self.logger = configure_logging(name=__name__)

    def register_model(
        self,
        model_path: str | Path,
        *,
        version: str | None = None,
        training_set_size: int | None = None,
        validation_metrics: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
        copy: bool = True,
    ) -> ModelVersion:
        """Register a model artifact and persist metadata."""

        source = Path(model_path)
        with self._locked_index(exclusive=True):
            index = self._read_index_unlocked()
            resolved_version = version or self._next_version_from_index(index)
            target = self.root_dir / f"{resolved_version}{source.suffix or '.pth'}"
            if copy:
                if not source.is_file():
                    msg = f"Cannot register missing model artifact: {source}"
                    raise FileNotFoundError(msg)
                shutil.copy2(source, target)
            else:
                target = source
            model = ModelVersion(
                version=resolved_version,
                path=target,
                created_at=datetime.now(timezone.utc).isoformat(),
                training_set_size=training_set_size,
                validation_metrics=dict(validation_metrics or {}),
                metadata=dict(metadata or {}),
            )
            index["models"] = [item for item in index["models"] if item["version"] != resolved_version]
            index["models"].append(self._to_json(model))
            self._write_index_unlocked(index)
            return model

    def deploy(
        self,
        *,
        version: str | None = None,
        config: dict[str, Any] | None = None,
        inference: Any | None = None,
        smoke_test: SmokeTest | None = None,
    ) -> ModelVersion:
        """Mark a version as deployed and optionally hot-reload inference."""

        model = self.get(version) if version is not None else self.latest()
        (smoke_test or default_smoke_test)(model.path)
        with self._locked_index(exclusive=True):
            index = self._read_index_unlocked()
            index["deployed_version"] = model.version
            self._write_index_unlocked(index)
        if config is not None:
            config.setdefault("allegro", {})["deployed_model_paths"] = [str(model.path)]
        if inference is not None and hasattr(inference, "reload"):
            inference.reload([model.path])
        self.logger.info("deployed model version=%s path=%s", model.version, model.path)
        return model

    def rollback(
        self,
        *,
        version: str,
        config: dict[str, Any] | None = None,
        inference: Any | None = None,
        smoke_test: SmokeTest | None = None,
    ) -> ModelVersion:
        """Deploy a previous version."""

        return self.deploy(version=version, config=config, inference=inference, smoke_test=smoke_test)

    def latest(self) -> ModelVersion:
        """Return the newest registered model."""

        models = self.list_models()
        if not models:
            msg = "No models are registered."
            raise ValueError(msg)
        return models[-1]

    def get(self, version: str) -> ModelVersion:
        """Return a registered version by id."""

        for model in self.list_models():
            if model.version == version:
                return model
        msg = f"Unknown model version: {version}"
        raise KeyError(msg)

    def list_models(self) -> list[ModelVersion]:
        """List models in registration order."""

        return [self._from_json(item) for item in self._load_index()["models"]]

    def next_version(self) -> str:
        """Return the next ``allegro_vNNN`` version string."""

        numbers = []
        for item in self._load_index()["models"]:
            version = item["version"]
            if version.startswith("allegro_v"):
                try:
                    numbers.append(int(version.removeprefix("allegro_v")))
                except ValueError:
                    continue
        return f"allegro_v{(max(numbers) + 1) if numbers else 1:03d}"

    def deployed_version(self) -> str | None:
        """Return the currently deployed version id, if any."""

        return self._load_index().get("deployed_version")

    def _load_index(self) -> dict[str, Any]:
        with self._locked_index(exclusive=False):
            return self._read_index_unlocked()

    def _write_index(self, index: dict[str, Any]) -> None:
        with self._locked_index(exclusive=True):
            self._write_index_unlocked(index)

    def _read_index_unlocked(self) -> dict[str, Any]:
        if not self.index_path.exists():
            return {"models": [], "deployed_version": None}
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    def _write_index_unlocked(self, index: dict[str, Any]) -> None:
        self.index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")

    def _next_version_from_index(self, index: dict[str, Any]) -> str:
        numbers = []
        for item in index["models"]:
            version = item["version"]
            if version.startswith("allegro_v"):
                try:
                    numbers.append(int(version.removeprefix("allegro_v")))
                except ValueError:
                    continue
        return f"allegro_v{(max(numbers) + 1) if numbers else 1:03d}"

    @contextmanager
    def _locked_index(self, *, exclusive: bool, timeout: float = 5.0) -> Iterator[None]:
        self.lock_path.touch(exist_ok=True)
        with self.lock_path.open("r+", encoding="utf-8") as lock_file:
            deadline = time.monotonic() + timeout
            while True:
                try:
                    if "fcntl" in globals():
                        mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
                        fcntl.flock(lock_file.fileno(), mode | fcntl.LOCK_NB)
                    else:
                        lock_file.seek(0)
                        mode = msvcrt.LK_NBLCK if exclusive else msvcrt.LK_NBRLCK
                        msvcrt.locking(lock_file.fileno(), mode, 1)
                    break
                except (BlockingIOError, OSError) as exc:
                    if time.monotonic() >= deadline:
                        msg = "Timed out waiting for model registry lock."
                        raise RuntimeError(msg) from exc
                    time.sleep(0.05)
            try:
                yield
            finally:
                if "fcntl" in globals():
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                else:
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)

    def _to_json(self, model: ModelVersion) -> dict[str, Any]:
        return {
            "version": model.version,
            "path": str(model.path),
            "created_at": model.created_at,
            "training_set_size": model.training_set_size,
            "validation_metrics": model.validation_metrics,
            "metadata": model.metadata,
        }

    def _from_json(self, data: dict[str, Any]) -> ModelVersion:
        return ModelVersion(
            version=data["version"],
            path=Path(data["path"]),
            created_at=data["created_at"],
            training_set_size=data.get("training_set_size"),
            validation_metrics=dict(data.get("validation_metrics") or {}),
            metadata=dict(data.get("metadata") or {}),
        )


def default_smoke_test(model_path: Path) -> None:
    """Basic deploy guard for model artifacts."""

    if not model_path.is_file():
        msg = f"Model artifact does not exist: {model_path}"
        raise FileNotFoundError(msg)
    if model_path.stat().st_size <= 0:
        msg = f"Model artifact is empty: {model_path}"
        raise ValueError(msg)
