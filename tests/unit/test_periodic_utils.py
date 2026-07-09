"""Tests for periodic boundary helper functions."""

from __future__ import annotations

import numpy as np
import pytest

from hotspot_al.utils.geometry import distances_to_group
from hotspot_al.utils.periodic import as_cell_matrix, mic_displacement, mic_displacements_from_reference


def _assert_vectorized_matches_loop(
    reference: np.ndarray,
    positions: np.ndarray,
    cell: np.ndarray | None,
    pbc: bool | tuple[bool, bool, bool],
) -> None:
    vectorized = mic_displacements_from_reference(reference, positions, cell=cell, pbc=pbc)
    looped = np.vstack([mic_displacement(reference, position, cell=cell, pbc=pbc) for position in positions])
    assert np.allclose(vectorized, looped, atol=1.0e-12)


def test_mic_displacements_from_reference_nonperiodic_random() -> None:
    rng = np.random.default_rng(123)
    reference = rng.normal(size=3)
    positions = rng.normal(size=(100, 3))

    displacements = mic_displacements_from_reference(reference, positions, pbc=False)

    assert np.allclose(displacements, positions - reference)


def test_mic_displacements_from_reference_orthogonal_pbc() -> None:
    rng = np.random.default_rng(123)
    reference = np.array([12.0, -1.0, 5.0])
    positions = rng.uniform(-5.0, 15.0, size=(100, 3))

    _assert_vectorized_matches_loop(reference, positions, np.diag([10.0, 10.0, 10.0]), True)


def test_mic_displacements_from_reference_non_orthogonal() -> None:
    rng = np.random.default_rng(456)
    reference = np.array([11.0, -3.0, 5.0])
    positions = rng.uniform(-8.0, 18.0, size=(100, 3))
    cell = np.array([[10.0, 2.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 12.0]])

    _assert_vectorized_matches_loop(reference, positions, cell, (True, True, False))


def test_mic_displacements_from_reference_partial_pbc() -> None:
    reference = np.array([9.5, 9.5, 0.0])
    positions = np.array([[0.5, 0.5, 20.0]])

    displacement = mic_displacements_from_reference(reference, positions, cell=np.diag([10.0, 10.0, 10.0]), pbc=[True, True, False])

    assert np.allclose(displacement[0], [1.0, 1.0, 20.0])


def test_as_cell_matrix_1d() -> None:
    assert np.allclose(as_cell_matrix(np.array([10.0, 8.0, 6.0])), np.diag([10.0, 8.0, 6.0]))


def test_as_cell_matrix_invalid_shape() -> None:
    with pytest.raises(ValueError, match="Unsupported cell shape"):
        as_cell_matrix(np.ones((2, 2)))


def test_distances_to_group_nonorthogonal_matches_loop() -> None:
    rng = np.random.default_rng(321)
    positions = rng.uniform(-2.0, 12.0, size=(40, 3))
    group_indices = [0, 7, 13]
    cell = np.array([[10.0, 1.5, 0.2], [0.0, 8.0, 0.7], [0.0, 0.0, 12.0]])
    pbc = [True, True, False]

    values = distances_to_group(positions, group_indices, cell=cell, pbc=pbc)
    expected = np.min(
        [
            [np.linalg.norm(mic_displacement(positions[index], position, cell=cell, pbc=pbc)) for position in positions]
            for index in group_indices
        ],
        axis=0,
    )

    assert np.allclose(values, expected, atol=1.0e-12)
