"""Geometry-based atom-wise monitors."""

from __future__ import annotations

import numpy as np
from ase import Atoms

from hotspot_al.utils.geometry import row_norms
from hotspot_al.utils.periodic import mic_displacement, mic_displacements_from_reference
from hotspot_al.monitor.neighbor_utils import MonitorNeighbors


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
    displacements = np.vstack(
        [mic_displacement(previous[i], current[i], cell=cell, pbc=pbc) for i in range(len(current))]
    )
    return row_norms(displacements)


def minimum_neighbor_distances(atoms: Atoms) -> np.ndarray:
    """Return the nearest-neighbor distance for each atom."""

    positions = atoms.get_positions()
    cell = atoms.cell.array
    pbc = atoms.pbc
    minima = np.full(len(atoms), np.inf, dtype=float)
    for index, position in enumerate(positions):
        displacements = mic_displacements_from_reference(position, positions, cell=cell, pbc=pbc)
        distances = row_norms(displacements)
        distances[index] = np.inf
        minima[index] = float(np.min(distances))
    minima[np.isinf(minima)] = 0.0
    return minima


def minimum_neighbor_distances_fast(atoms: Atoms, nl: MonitorNeighbors | None = None) -> np.ndarray:
    """Return nearest-neighbor distances with an optional neighbor list."""

    if nl is None:
        return minimum_neighbor_distances(atoms)

    minima = np.full(len(atoms), np.inf, dtype=float)
    for index in range(len(atoms)):
        _indices, _displacements, distances = nl.get_displacements(atoms, index, nl.cutoff)
        if len(distances):
            minima[index] = float(np.min(distances))
    minima[np.isinf(minima)] = 0.0
    return minima
