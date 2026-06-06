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
