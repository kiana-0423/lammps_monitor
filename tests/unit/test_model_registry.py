"""Tests for Allegro model registry helpers."""

from __future__ import annotations

from pathlib import Path
from threading import Thread

import pytest

from hotspot_al.training.model_registry import ModelRegistry


class _FakeInference:
    def __init__(self) -> None:
        self.paths = []

    def reload(self, model_paths) -> None:
        self.paths = list(model_paths)


def test_model_registry_registers_deploys_and_rolls_back(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    model_a = tmp_path / "a.pth"
    model_b = tmp_path / "b.pth"
    model_a.write_text("a", encoding="utf-8")
    model_b.write_text("b", encoding="utf-8")

    first = registry.register_model(model_a, training_set_size=2)
    second = registry.register_model(model_b, validation_metrics={"mae": 0.1})
    config = {"allegro": {}}
    inference = _FakeInference()

    deployed = registry.deploy(config=config, inference=inference)
    rolled_back = registry.rollback(version=first.version, config=config, inference=inference)

    assert first.version == "allegro_v001"
    assert second.version == "allegro_v002"
    assert deployed.version == second.version
    assert rolled_back.version == first.version
    assert config["allegro"]["deployed_model_paths"] == [str(first.path)]
    assert inference.paths == [first.path]


def test_model_registry_default_smoke_test_rejects_empty_model(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    model = tmp_path / "empty.pth"
    model.write_text("", encoding="utf-8")
    registered = registry.register_model(model)

    with pytest.raises(ValueError, match="empty"):
        registry.deploy(version=registered.version)


def test_model_registry_concurrent_writes(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    model = tmp_path / "model.pth"
    model.write_text("model", encoding="utf-8")
    errors: list[BaseException] = []

    def register(version: str) -> None:
        try:
            registry.register_model(model, version=version, copy=False)
        except BaseException as exc:
            errors.append(exc)

    threads = [
        Thread(target=register, args=(f"allegro_v{index:03d}",))
        for index in range(1, 6)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    versions = {model.version for model in registry.list_models()}
    assert versions == {f"allegro_v{index:03d}" for index in range(1, 6)}
