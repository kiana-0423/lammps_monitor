"""Neighbor and bond inference utilities."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from ase.data import covalent_radii

from .periodic import mic_displacement


def infer_bonds(
    positions: np.ndarray,
    numbers: np.ndarray,
    cell: np.ndarray | None = None,
    pbc: bool | tuple[bool, bool, bool] | np.ndarray = False,
    scale: float = 1.2,
) -> list[tuple[int, int]]:
    """Infer covalent bonds from geometry using scaled covalent radii."""

    positions = np.asarray(positions, dtype=float)
    numbers = np.asarray(numbers, dtype=int)
    bonds: list[tuple[int, int]] = []
    for i in range(len(positions)):
        radius_i = covalent_radii[numbers[i]]
        for j in range(i + 1, len(positions)):
            radius_j = covalent_radii[numbers[j]]
            cutoff = scale * (radius_i + radius_j)
            distance = np.linalg.norm(mic_displacement(positions[i], positions[j], cell=cell, pbc=pbc))
            if distance <= cutoff:
                bonds.append((i, j))
    return bonds


def bonded_neighbors(
    positions: np.ndarray,
    numbers: np.ndarray,
    cell: np.ndarray | None = None,
    pbc: bool | tuple[bool, bool, bool] | np.ndarray = False,
    scale: float = 1.2,
) -> dict[int, list[int]]:
    """Return an adjacency list inferred from covalent bond criteria."""

    adjacency: dict[int, list[int]] = defaultdict(list)
    for i, j in infer_bonds(positions, numbers, cell=cell, pbc=pbc, scale=scale):
        adjacency[i].append(j)
        adjacency[j].append(i)
    return dict(adjacency)
