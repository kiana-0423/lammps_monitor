"""Neighbor and bond inference utilities."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.neighborlist import neighbor_list


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
    if len(positions) < 2:
        return []

    radii = covalent_radii[numbers]
    max_cutoff = float(scale * 2.0 * np.max(radii))
    use_pbc = pbc if cell is not None else False
    atoms = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=use_pbc)
    left, right, distances = neighbor_list("ijd", atoms, max_cutoff)
    unique_mask = left < right
    left = left[unique_mask]
    right = right[unique_mask]
    distances = distances[unique_mask]
    cutoffs = scale * (radii[left] + radii[right])
    bonded_mask = distances <= cutoffs
    return list(zip(left[bonded_mask].astype(int).tolist(), right[bonded_mask].astype(int).tolist()))


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
