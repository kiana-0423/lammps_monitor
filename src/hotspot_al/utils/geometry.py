"""Geometry utilities shared by monitors and extraction modules."""

from __future__ import annotations

import numpy as np

from .periodic import mic_displacements_from_reference


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
    for index in group_indices:
        displacements = mic_displacements_from_reference(
            positions[index],
            positions,
            cell=cell,
            pbc=pbc,
        )
        distances = np.minimum(distances, row_norms(displacements))
    return distances


def padded_cluster_cell(positions: np.ndarray, padding: float = 4.0) -> np.ndarray:
    """Create a non-periodic orthorhombic cell around positions."""

    positions = np.asarray(positions, dtype=float)
    mins = positions.min(axis=0) - padding
    maxs = positions.max(axis=0) + padding
    lengths = np.maximum(maxs - mins, padding)
    return np.diag(lengths)
