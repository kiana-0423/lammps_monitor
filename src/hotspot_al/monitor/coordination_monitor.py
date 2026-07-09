"""Smooth coordination number monitors."""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.neighborlist import neighbor_list

from hotspot_al.monitor.neighbor_utils import MonitorNeighbors
from hotspot_al.utils.periodic import mic_displacements_from_reference

_BATCH_NEIGHBOR_MIN_ATOMS = 256


def smooth_coordination_numbers(
    atoms: Atoms,
    *,
    scale: float = 1.15,
    power: int = 6,
) -> np.ndarray:
    """Compute smooth coordination numbers for all atoms."""

    positions = atoms.get_positions()
    numbers = atoms.get_atomic_numbers()
    cell = atoms.cell.array
    pbc = atoms.pbc
    q_values = np.zeros(len(atoms), dtype=float)
    for i, position in enumerate(positions):
        displacements = mic_displacements_from_reference(position, positions, cell=cell, pbc=pbc)
        distances = np.linalg.norm(displacements, axis=1)
        distances[i] = np.inf
        radii = scale * (covalent_radii[numbers[i]] + covalent_radii[numbers])
        weights = 1.0 / (1.0 + np.power(distances / radii, power))
        weights[i] = 0.0
        q_values[i] = float(np.sum(weights))
    return q_values


def smooth_coordination_numbers_fast(
    atoms: Atoms,
    nl: MonitorNeighbors | None = None,
    *,
    scale: float = 1.15,
    power: int = 6,
    cutoff: float | None = None,
) -> np.ndarray:
    """Compute smooth coordination numbers with an optional neighbor list."""

    if nl is None:
        return smooth_coordination_numbers(atoms, scale=scale, power=power)

    numbers = atoms.get_atomic_numbers()
    q_values = np.zeros(len(atoms), dtype=float)
    sub_cutoff = nl.coordination_cutoff if cutoff is None else float(cutoff)
    if len(atoms) < _BATCH_NEIGHBOR_MIN_ATOMS:
        for index in range(len(atoms)):
            indices, _displacements, distances = nl.get_displacements(atoms, index, sub_cutoff)
            if len(indices) == 0:
                continue
            radii = scale * (covalent_radii[numbers[index]] + covalent_radii[numbers[indices]])
            weights = 1.0 / (1.0 + np.power(distances / radii, power))
            q_values[index] = float(np.sum(weights))
        return q_values

    left, right, distances = neighbor_list("ijd", atoms, sub_cutoff)
    if len(left) == 0:
        return q_values
    mask = distances > 1.0e-8
    left = left[mask]
    right = right[mask]
    distances = distances[mask]
    radii = scale * (covalent_radii[numbers[left]] + covalent_radii[numbers[right]])
    weights = 1.0 / (1.0 + np.power(distances / radii, power))
    np.add.at(q_values, left, weights)
    return q_values


def coordination_deltas(current_q: np.ndarray, previous_q: np.ndarray | None) -> np.ndarray:
    """Return per-atom coordination changes between frames."""

    current = np.asarray(current_q, dtype=float)
    if previous_q is None:
        return np.zeros(len(current), dtype=float)
    return np.abs(current - np.asarray(previous_q, dtype=float))
