"""Tests for staged OOD score metric activation."""

from __future__ import annotations

import numpy as np

from hotspot_al.config import load_config
from hotspot_al.monitor.ood_score import OODScorer


def test_physics_stage_includes_mlip_force_deviation() -> None:
    config = load_config()
    config["ood_score"] = {
        **config["ood_score"],
        "weights": {key: 0.0 for key in config["ood_score"]["weights"]},
        "physics_threshold": 2.0,
        "running_stats": {"enabled": True, "warmup_frames": 0, "min_std": 1.0},
    }
    config["ood_score"]["weights"]["mlip_force_deviation"] = 1.0
    metrics = {
        "force": np.zeros(2),
        "delta_force": np.zeros(2),
        "rmin": np.ones(2),
        "delta_q": np.zeros(2),
        "displacement": np.zeros(2),
        "mlip_force_deviation": np.array([3.0, 0.0]),
    }

    result = OODScorer(config).score_physics(metrics, update_stats=False)

    assert result.triggered
    assert "mlip_force_deviation" in result.metric_scores
    assert result.hotspot_indices == [0]


def _base_metrics(n_atoms: int = 2) -> dict[str, np.ndarray]:
    return {
        "force": np.zeros(n_atoms),
        "delta_force": np.zeros(n_atoms),
        "rmin": np.ones(n_atoms),
        "delta_q": np.zeros(n_atoms),
        "displacement": np.zeros(n_atoms),
        "mlip_force_deviation": np.zeros(n_atoms),
        "lj_residual": np.zeros(n_atoms),
        "committee": np.zeros(n_atoms),
    }


def _score_config(*, min_trigger_atoms: int = 1) -> dict:
    return {
        "monitor": {"delta_q_threshold": 10.0, "displacement_z_threshold": 10.0},
        "ood_score": {
            "weights": {
                "force": 1.0,
                "delta_force": 0.0,
                "rmin": 0.0,
                "delta_q": 0.0,
                "lj_residual": 0.0,
                "committee": 0.0,
                "displacement": 0.0,
                "mlip_force_deviation": 1.0,
            },
            "screen_threshold": 4.0,
            "physics_threshold": 5.0,
            "label_threshold": 6.0,
            "lj_lazy_threshold": 100.0,
            "min_trigger_atoms": min_trigger_atoms,
            "running_stats": {"enabled": True, "warmup_frames": 5, "min_std": 1.0},
        },
    }


def test_score_light_no_trigger_below_screen_threshold() -> None:
    config = _score_config()
    metrics = _base_metrics()
    metrics["force"] = np.array([1.0, 3.9])

    result = OODScorer(config).score_light(metrics, update_stats=False)

    assert not result.triggered
    assert result.hotspot_indices == []


def test_score_full_triggers_on_mlip_force_deviation() -> None:
    config = _score_config()
    metrics = _base_metrics()
    metrics["mlip_force_deviation"] = np.array([6.5, 0.0])

    result = OODScorer(config).score_full(metrics, update_stats=False)

    assert result.triggered
    assert result.hotspot_indices == [0]
    assert result.trigger_reason == ["model_drift"]


def test_running_stats_warmup_uses_raw_scores() -> None:
    config = _score_config()
    metrics = _base_metrics()
    metrics["force"] = np.array([2.5, 0.5])

    result = OODScorer(config).score_light(metrics, update_stats=False)

    assert np.allclose(result.metric_scores["force"], np.array([2.5, 0.5]))


def test_min_trigger_atoms_enforced() -> None:
    config = _score_config(min_trigger_atoms=3)
    metrics = _base_metrics(n_atoms=3)
    metrics["force"] = np.array([7.0, 0.0, 0.0])

    result = OODScorer(config).score_light(metrics, update_stats=False)

    assert not result.triggered


def test_empty_frame_returns_no_trigger() -> None:
    config = _score_config()
    metrics = _base_metrics(n_atoms=0)

    result = OODScorer(config).score_full(metrics, update_stats=False)

    assert not result.triggered
    assert result.hotspot_indices == []
    assert result.max_score == 0.0
