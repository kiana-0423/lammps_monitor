"""Geometry-based atom-wise monitors."""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list

from hotspot_al.monitor.neighbor_utils import MonitorNeighbors
from hotspot_al.utils.geometry import row_norms
from hotspot_al.utils.periodic import as_cell_matrix

_PAIRWISE_DISTANCE_CHUNK = 512
_BATCH_NEIGHBOR_MIN_ATOMS = 512


def displacement_norms(
    current_positions: np.ndarray,
    previous_positions: np.ndarray | None,
    *,
    cell: np.ndarray | None = None,
    pbc: bool | tuple[bool, bool, bool] | np.ndarray = False,
) -> np.ndarray:
    """Return per-atom single-step displacements."""

    current = np.asarray(current_positions, dtype=float)
    if previous_positions is None:
        return np.zeros(len(current), dtype=float)
    previous = np.asarray(previous_positions, dtype=float)
    displacements = current - previous
    cell_matrix = as_cell_matrix(cell)
    pbc_mask = np.broadcast_to(np.asarray(pbc, dtype=bool), 3)
    if cell_matrix is not None and np.any(pbc_mask):
        inverse = np.linalg.inv(cell_matrix.T)
        fractional = displacements @ inverse.T
        fractional[:, pbc_mask] -= np.round(fractional[:, pbc_mask])
        displacements = fractional @ cell_matrix
    return row_norms(displacements)


def minimum_neighbor_distances(atoms: Atoms) -> np.ndarray:
    """Return the nearest-neighbor distance for each atom."""

    positions = atoms.get_positions()
    cell = atoms.cell.array
    pbc = atoms.pbc
    n_atoms = len(atoms)
    minima = np.full(n_atoms, np.inf, dtype=float)
    if n_atoms <= 1:
        return np.zeros(n_atoms, dtype=float)
    cell_matrix = as_cell_matrix(cell)
    pbc_mask = np.broadcast_to(np.asarray(pbc, dtype=bool), 3)
    inverse = np.linalg.inv(cell_matrix.T) if cell_matrix is not None and np.any(pbc_mask) else None
    for start in range(0, n_atoms, _PAIRWISE_DISTANCE_CHUNK):
        stop = min(start + _PAIRWISE_DISTANCE_CHUNK, n_atoms)
        displacements = positions[None, :, :] - positions[start:stop, None, :]
        if inverse is not None and cell_matrix is not None:
            fractional = displacements @ inverse.T
            fractional[:, :, pbc_mask] -= np.round(fractional[:, :, pbc_mask])
            displacements = fractional @ cell_matrix
        distances = np.linalg.norm(displacements, axis=2)
        distances[np.arange(stop - start), np.arange(start, stop)] = np.inf
        minima[start:stop] = np.min(distances, axis=1)
    minima[np.isinf(minima)] = 0.0
    return minima


def minimum_neighbor_distances_fast(atoms: Atoms, nl: MonitorNeighbors | None = None) -> np.ndarray:
    """Return nearest-neighbor distances with an optional neighbor list."""

    if nl is None:
        return minimum_neighbor_distances(atoms)

    if len(atoms) < _BATCH_NEIGHBOR_MIN_ATOMS:
        return minimum_neighbor_distances(atoms)

    minima = np.full(len(atoms), np.inf, dtype=float)
    left, distances = neighbor_list("id", atoms, nl.cutoff)
    if len(left):
        mask = distances > 1.0e-8
        np.minimum.at(minima, left[mask], distances[mask])
    minima[np.isinf(minima)] = 0.0
    return minima
