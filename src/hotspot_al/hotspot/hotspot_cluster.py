"""Spatial clustering of anomalous atoms into hotspot regions."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.spatial.distance import pdist, squareform

from hotspot_al.utils.periodic import mic_displacements_from_reference


def cluster_anomalous_atoms(
    flagged_indices: list[int],
    positions: np.ndarray,
    *,
    merge_radius: float,
    cell: np.ndarray | None = None,
    pbc: bool | tuple[bool, bool, bool] | np.ndarray = False,
) -> list[list[int]]:
    """Merge anomalous atoms into connected hotspot clusters."""

    if not flagged_indices:
        return []
    n_flagged = len(flagged_indices)
    flagged_positions = np.asarray(positions, dtype=float)[flagged_indices]
    pbc_mask = np.broadcast_to(np.asarray(pbc, dtype=bool), 3)
    if cell is None or not np.any(pbc_mask):
        distance_matrix = squareform(pdist(flagged_positions)) if n_flagged > 1 else np.zeros((n_flagged, n_flagged))
    else:
        distance_matrix = np.zeros((n_flagged, n_flagged), dtype=float)
        for row, position in enumerate(flagged_positions):
            displacements = mic_displacements_from_reference(position, flagged_positions, cell=cell, pbc=pbc)
            distance_matrix[row] = np.linalg.norm(displacements, axis=1)
    parent = {index: index for index in flagged_indices}

    def find(index: int) -> int:
        root = parent[index]
        while root != parent[root]:
            parent[root] = parent[parent[root]]
            root = parent[root]
        while index != root:
            next_index = parent[index]
            parent[index] = root
            index = next_index
        return root

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for i, left in enumerate(flagged_indices):
        for j in range(i + 1, n_flagged):
            if distance_matrix[i, j] <= merge_radius:
                union(left, flagged_indices[j])

    clusters: dict[int, list[int]] = defaultdict(list)
    for index in flagged_indices:
        clusters[find(index)].append(index)
    return [sorted(indices) for indices in clusters.values()]
