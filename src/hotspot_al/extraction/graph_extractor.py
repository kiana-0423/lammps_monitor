"""Graph-based extraction for molecular and polymeric environments."""

from __future__ import annotations

from collections import deque
from typing import Any

from ase import Atoms

from hotspot_al.extraction.cluster_extractor import extract_cluster_region
from hotspot_al.models import ExtractedRegion
from hotspot_al.utils.neighbor import bonded_neighbors


def extract_graph_region(
    atoms: Atoms,
    hotspot_indices: list[int],
    *,
    config: dict[str, Any],
) -> ExtractedRegion:
    """Extract a connected molecular subgraph around hotspot atoms.

    This baseline implementation expands a covalent bond graph out to a fixed
    number of hops, then falls back to the same radial region partitioning used
    by cluster extraction.
    """

    graph_cfg = config.get("extraction", {}).get("graph", {})
    hops = int(graph_cfg.get("hops", 2))
    adjacency = bonded_neighbors(
        atoms.get_positions(),
        atoms.get_atomic_numbers(),
        cell=atoms.cell.array,
        pbc=atoms.pbc,
        scale=float(config.get("h_capping", {}).get("covalent_scale", 1.2)),
    )

    visited = set(hotspot_indices)
    queue: deque[tuple[int, int]] = deque((index, 0) for index in hotspot_indices)
    while queue:
        node, depth = queue.popleft()
        if depth >= hops:
            continue
        for neighbor in adjacency.get(node, []):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append((neighbor, depth + 1))

    region = extract_cluster_region(atoms, sorted(visited), config=config)
    region.metadata = {**region.metadata, "mode": "graph", "graph_hops": hops}
    return region
