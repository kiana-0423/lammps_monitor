"""Deterministic fake Allegro evaluators for offline tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms


def _model_scale(model_path: str | Path | None) -> float:
    if model_path is None:
        return 1.0
    text = str(model_path)
    return float((sum(ord(char) for char in text) % 7) + 1)


def fake_force_evaluator(atoms: Atoms, model_path: str | Path | None, config: dict[str, Any]) -> np.ndarray:
    """Return deterministic forces with shape ``(n_atoms, 3)``."""

    positions = atoms.get_positions()
    scale = _model_scale(model_path) * float(config.get("fake_allegro_scale", 0.01))
    return scale * (positions - positions.mean(axis=0, keepdims=True))


def fake_committee_evaluator(atoms: Atoms, model_paths: list[str | Path], config: dict[str, Any]) -> np.ndarray:
    """Return deterministic committee forces with shape ``(n_models, n_atoms, 3)``."""

    if not model_paths:
        msg = "fake_committee_evaluator requires at least one model path."
        raise ValueError(msg)
    return np.stack([fake_force_evaluator(atoms, model_path, config) for model_path in model_paths], axis=0)

