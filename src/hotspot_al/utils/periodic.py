"""Periodic boundary condition helpers."""

from __future__ import annotations

import numpy as np


def as_cell_matrix(cell: np.ndarray | None) -> np.ndarray | None:
    """Normalize a cell description to a 3x3 matrix."""

    if cell is None:
        return None
    matrix = np.asarray(cell, dtype=float)
    if matrix.shape == (3,):
        return np.diag(matrix)
    if matrix.shape != (3, 3):
        msg = f"Unsupported cell shape: {matrix.shape}"
        raise ValueError(msg)
    return matrix


def mic_displacement(
    source: np.ndarray,
    target: np.ndarray,
    cell: np.ndarray | None = None,
    pbc: bool | tuple[bool, bool, bool] | np.ndarray = False,
) -> np.ndarray:
    """Return the minimum-image displacement from ``source`` to ``target``."""

    displacement = np.asarray(target, dtype=float) - np.asarray(source, dtype=float)
    cell_matrix = as_cell_matrix(cell)
    pbc_mask = np.broadcast_to(np.asarray(pbc, dtype=bool), 3)
    if cell_matrix is None or not np.any(pbc_mask):
        return displacement

    inverse = np.linalg.inv(cell_matrix.T)
    fractional = inverse @ displacement
    fractional[pbc_mask] -= np.round(fractional[pbc_mask])
    return cell_matrix.T @ fractional


def mic_displacements_from_reference(
    reference: np.ndarray,
    positions: np.ndarray,
    cell: np.ndarray | None = None,
    pbc: bool | tuple[bool, bool, bool] | np.ndarray = False,
) -> np.ndarray:
    """Return minimum-image displacements from one reference to many positions."""

    reference = np.asarray(reference, dtype=float)
    positions = np.asarray(positions, dtype=float)
    return np.vstack([mic_displacement(reference, pos, cell=cell, pbc=pbc) for pos in positions])


def mic_distance(
    source: np.ndarray,
    target: np.ndarray,
    cell: np.ndarray | None = None,
    pbc: bool | tuple[bool, bool, bool] | np.ndarray = False,
) -> float:
    """Return the minimum-image distance between two points."""

    return float(np.linalg.norm(mic_displacement(source, target, cell=cell, pbc=pbc)))
