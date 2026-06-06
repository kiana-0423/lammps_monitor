"""Tests for neighbor-list and linear LJ monitor fast paths."""

from __future__ import annotations

import numpy as np
from ase import Atoms

from hotspot_al.monitor.coordination_monitor import smooth_coordination_numbers, smooth_coordination_numbers_fast
from hotspot_al.monitor.geometry_monitor import minimum_neighbor_distances, minimum_neighbor_distances_fast
from hotspot_al.monitor.lj_residual_fast import fit_local_lj_force_linear
from hotspot_al.monitor.neighbor_utils import MonitorNeighbors
from hotspot_al.monitor.ood_score import OODScorer


def test_neighbor_fast_paths_match_full_search_when_cutoff_covers_system() -> None:
    atoms = Atoms(
        "H10",
        positions=np.array([[0.7 * i, 0.1 * (i % 2), 0.0] for i in range(10)], dtype=float),
        cell=[30.0, 30.0, 30.0],
        pbc=True,
    )
    neighbors = MonitorNeighbors(atoms, lj_cutoff=30.0, coordination_cutoff=30.0)

    np.testing.assert_allclose(
        minimum_neighbor_distances_fast(atoms, neighbors),
        minimum_neighbor_distances(atoms),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        smooth_coordination_numbers_fast(atoms, neighbors),
        smooth_coordination_numbers(atoms),
        rtol=1.0e-6,
        atol=1.0e-8,
    )


def test_linear_lj_fit_recovers_known_parameters() -> None:
    vectors = np.array(
        [
            [1.2, 0.0, 0.0],
            [0.0, 1.4, 0.0],
            [0.0, 0.0, 1.6],
            [-1.8, 0.3, 0.0],
        ],
        dtype=float,
    )
    epsilon = 0.35
    sigma = 1.1
    distances = np.linalg.norm(vectors, axis=1)
    target = np.sum(
        24.0
        * epsilon
        * ((2.0 * sigma**12 / distances**14) - (sigma**6 / distances**8))[:, None]
        * vectors,
        axis=0,
    )

    fit = fit_local_lj_force_linear(vectors, target)

    assert fit.valid
    assert fit.residual < 1.0e-10
    np.testing.assert_allclose(fit.epsilon, epsilon, rtol=1.0e-10, atol=1.0e-12)
    np.testing.assert_allclose(fit.sigma, sigma, rtol=1.0e-10, atol=1.0e-12)


def test_score_physics_lazy_lj_only_fits_suspicious_atoms() -> None:
    atoms = Atoms(
        "H4",
        positions=[
            [0.0, 0.0, 0.0],
            [1.2, 0.0, 0.0],
            [0.0, 1.3, 0.0],
            [0.0, 0.0, 1.4],
        ],
        cell=[10.0, 10.0, 10.0],
        pbc=False,
    )
    config = {
        "monitor": {"lj_cutoff": 4.0, "delta_q_threshold": 10.0, "displacement_z_threshold": 10.0},
        "ood_score": {
            "weights": {
                "force": 1.0,
                "delta_force": 0.0,
                "rmin": 0.0,
                "delta_q": 0.0,
                "lj_residual": 1.0,
                "committee": 0.0,
                "displacement": 0.0,
            },
            "screen_threshold": 4.0,
            "physics_threshold": 5.0,
            "label_threshold": 6.0,
            "lj_lazy_threshold": 3.0,
            "running_stats": {"warmup_frames": 0, "min_std": 1.0},
        },
    }
    metrics = {
        "force": np.array([0.0, 5.0, 0.0, 0.0]),
        "delta_force": np.zeros(4),
        "rmin": np.ones(4),
        "delta_q": np.zeros(4),
        "displacement": np.zeros(4),
    }
    forces = np.zeros((4, 3), dtype=float)

    result = OODScorer(config).score_physics(metrics, atoms=atoms, forces=forces, update_stats=False)

    assert result.metadata["lj_fit_count"] == 1
    assert result.metric_scores["lj_residual"][0] == 0.0
    assert result.metric_scores["lj_residual"][2] == 0.0
    assert result.metric_scores["lj_residual"][3] == 0.0
