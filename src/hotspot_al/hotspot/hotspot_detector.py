"""Detect hotspot clusters from atom-wise OOD scores."""

from __future__ import annotations

import numpy as np
from ase import Atoms

from hotspot_al.hotspot.hotspot_cluster import cluster_anomalous_atoms
from hotspot_al.models import Hotspot
from hotspot_al.utils.geometry import hotspot_center


def detect_hotspots(
    atoms: Atoms,
    atom_scores: np.ndarray,
    *,
    threshold: float,
    merge_radius: float = 8.0,
    step: int = 0,
    trigger_reasons: list[str] | None = None,
    event_id: str | None = None,
    backend: str | None = None,
) -> list[Hotspot]:
    """Detect merged hotspot regions from atom-wise anomaly scores."""

    scores = np.asarray(atom_scores, dtype=float)
    flagged = np.where(scores >= threshold)[0].tolist()
    if not flagged and len(scores):
        flagged = [int(np.argmax(scores))]
    positions = atoms.get_positions()
    clusters = cluster_anomalous_atoms(
        flagged,
        positions,
        merge_radius=merge_radius,
        cell=atoms.cell.array,
        pbc=atoms.pbc,
    )
    hotspots: list[Hotspot] = []
    for cluster in clusters:
        center = hotspot_center(positions, cluster, cell=atoms.cell.array, pbc=atoms.pbc)
        hotspots.append(
            Hotspot(
                core_atom_indices=cluster,
                center=center,
                max_score=float(np.max(scores[cluster])),
                trigger_reasons=list(trigger_reasons or []),
                step=step,
                event_id=event_id,
                backend=backend,
                metadata={"n_atoms": len(cluster)},
            )
        )
    hotspots.sort(key=lambda item: item.max_score, reverse=True)
    return hotspots
