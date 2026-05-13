"""Smooth coordination number monitors."""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.data import covalent_radii

from hotspot_al.utils.periodic import mic_displacements_from_reference


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


def coordination_deltas(current_q: np.ndarray, previous_q: np.ndarray | None) -> np.ndarray:
    """Return per-atom coordination changes between frames."""

    current = np.asarray(current_q, dtype=float)
    if previous_q is None:
        return np.zeros(len(current), dtype=float)
    return np.abs(current - np.asarray(previous_q, dtype=float))
