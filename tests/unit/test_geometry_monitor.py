"""Tests for geometry-based monitor helpers."""

from __future__ import annotations

import numpy as np

from hotspot_al.monitor.geometry_monitor import displacement_norms
from hotspot_al.utils.periodic import mic_displacement


def test_displacement_norms_nonperiodic_matches_plain_difference() -> None:
    rng = np.random.default_rng(7)
    previous = rng.normal(size=(1000, 3))
    current = previous + rng.normal(scale=0.1, size=(1000, 3))

    values = displacement_norms(current, previous, pbc=False)

    assert np.allclose(values, np.linalg.norm(current - previous, axis=1))


def test_displacement_norms_orthogonal_pbc_matches_loop() -> None:
    rng = np.random.default_rng(8)
    previous = rng.uniform(0.0, 10.0, size=(1000, 3))
    current = previous + rng.uniform(-6.0, 6.0, size=(1000, 3))
    cell = np.diag([10.0, 10.0, 10.0])

    values = displacement_norms(current, previous, cell=cell, pbc=True)
    expected = np.array(
        [np.linalg.norm(mic_displacement(previous[index], current[index], cell=cell, pbc=True)) for index in range(len(current))]
    )

    assert np.allclose(values, expected, atol=1.0e-12)


def test_displacement_norms_nonorthogonal_partial_pbc_matches_loop() -> None:
    rng = np.random.default_rng(12)
    previous = rng.uniform(-2.0, 12.0, size=(500, 3))
    current = previous + rng.uniform(-6.0, 6.0, size=(500, 3))
    cell = np.array([[10.0, 2.0, 0.0], [0.0, 8.0, 1.0], [0.0, 0.0, 12.0]])
    pbc = [True, True, False]

    values = displacement_norms(current, previous, cell=cell, pbc=pbc)
    expected = np.array(
        [
            np.linalg.norm(mic_displacement(previous[index], current[index], cell=cell, pbc=pbc))
            for index in range(len(current))
        ]
    )

    assert np.allclose(values, expected, atol=1.0e-12)
