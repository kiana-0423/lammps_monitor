"""Geometry utilities shared by monitors and extraction modules."""

from __future__ import annotations

import numpy as np

from .periodic import as_cell_matrix, mic_displacements_from_reference

_DISTANCE_MATRIX_GROUP_CHUNK = 128


def row_norms(values: np.ndarray) -> np.ndarray:
    """Compute vector norms row-wise."""

    array = np.asarray(values, dtype=float)
    return np.linalg.norm(array, axis=-1)


def hotspot_center(
    positions: np.ndarray,
    hotspot_indices: list[int],
    cell: np.ndarray | None = None,
    pbc: bool | tuple[bool, bool, bool] | np.ndarray = False,
) -> np.ndarray:
    """Compute a contiguous centroid for hotspot atoms under PBC."""

    if not hotspot_indices:
        msg = "Cannot compute a hotspot center for an empty index list."
        raise ValueError(msg)
    anchor = positions[hotspot_indices[0]]
    displacements = mic_displacements_from_reference(anchor, positions[hotspot_indices], cell=cell, pbc=pbc)
    return anchor + np.mean(displacements, axis=0)


def distances_to_group(
    positions: np.ndarray,
    group_indices: list[int],
    cell: np.ndarray | None = None,
    pbc: bool | tuple[bool, bool, bool] | np.ndarray = False,
) -> np.ndarray:
    """Return the minimum distance from each atom to a group of atoms."""

    positions = np.asarray(positions, dtype=float)
    distances = np.full(len(positions), np.inf, dtype=float)
    if not group_indices:
        return distances
    group = positions[np.asarray(group_indices, dtype=int)]
    cell_matrix = as_cell_matrix(cell)
    pbc_mask = np.broadcast_to(np.asarray(pbc, dtype=bool), 3)
    inverse = np.linalg.inv(cell_matrix.T) if cell_matrix is not None and np.any(pbc_mask) else None
    for start in range(0, len(group), _DISTANCE_MATRIX_GROUP_CHUNK):
        references = group[start : start + _DISTANCE_MATRIX_GROUP_CHUNK]
        displacements = positions[None, :, :] - references[:, None, :]
        if inverse is not None and cell_matrix is not None:
            fractional = displacements @ inverse.T
            fractional[:, :, pbc_mask] -= np.round(fractional[:, :, pbc_mask])
            displacements = fractional @ cell_matrix
        distances = np.minimum(distances, np.min(np.linalg.norm(displacements, axis=2), axis=0))
    return distances


def padded_cluster_cell(positions: np.ndarray, padding: float = 4.0) -> np.ndarray:
    """Create a non-periodic orthorhombic cell around positions."""

    positions = np.asarray(positions, dtype=float)
    mins = positions.min(axis=0) - padding
    maxs = positions.max(axis=0) + padding
    lengths = np.maximum(maxs - mins, padding)
    return np.diag(lengths)
