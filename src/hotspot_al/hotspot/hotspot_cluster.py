"""Spatial clustering of anomalous atoms into hotspot regions."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from hotspot_al.utils.periodic import mic_displacement


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
        for right in flagged_indices[i + 1 :]:
            distance = np.linalg.norm(mic_displacement(positions[left], positions[right], cell=cell, pbc=pbc))
            if distance <= merge_radius:
                union(left, right)

    clusters: dict[int, list[int]] = defaultdict(list)
    for index in flagged_indices:
        clusters[find(index)].append(index)
    return [sorted(indices) for indices in clusters.values()]
