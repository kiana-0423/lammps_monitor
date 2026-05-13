"""Cluster extraction around spatial hotspot regions."""

from __future__ import annotations

from typing import Any

import numpy as np
from ase import Atoms

from hotspot_al.extraction.boundary_regions import assign_radial_regions, region_label_vector
from hotspot_al.models import ExtractedRegion
from hotspot_al.utils.geometry import distances_to_group, hotspot_center, padded_cluster_cell
from hotspot_al.utils.periodic import mic_displacements_from_reference


def extract_cluster_region(
    atoms: Atoms,
    hotspot_indices: list[int],
    *,
    config: dict[str, Any],
    center: np.ndarray | None = None,
) -> ExtractedRegion:
    """Extract a non-periodic cluster centered on a hotspot."""

    if not hotspot_indices:
        msg = "At least one hotspot atom is required for cluster extraction."
        raise ValueError(msg)

    extraction_cfg = config.get("extraction", config)
    positions = atoms.get_positions()
    center = hotspot_center(positions, hotspot_indices, cell=atoms.cell.array, pbc=atoms.pbc) if center is None else center
    distances = distances_to_group(positions, hotspot_indices, cell=atoms.cell.array, pbc=atoms.pbc)

    extract_radius = float(extraction_cfg.get("extract_radius", extraction_cfg.get("mlip_cutoff", 6.0) + extraction_cfg.get("buffer_radius", 4.0)))
    max_atoms = int(extraction_cfg.get("max_atoms", len(atoms)))
    min_atoms = int(extraction_cfg.get("min_atoms", min(len(atoms), len(hotspot_indices))))
    vacuum_padding = float(extraction_cfg.get("vacuum_padding", 4.0))

    selected = np.where(distances <= extract_radius)[0].tolist()
    if len(selected) < min_atoms:
        sorted_by_distance = np.argsort(distances).tolist()
        selected = sorted(sorted_by_distance[: min(min_atoms, len(atoms))])
    if len(selected) > max_atoms:
        sorted_by_distance = np.argsort(distances[selected])
        selected = sorted([selected[i] for i in sorted_by_distance[:max_atoms]])

    selected_positions = positions[selected]
    displacements = mic_displacements_from_reference(center, selected_positions, cell=atoms.cell.array, pbc=atoms.pbc)
    local_positions = displacements
    mins = local_positions.min(axis=0)
    shift = -mins + vacuum_padding * 0.5
    cluster_positions = local_positions + shift
    cluster_cell = padded_cluster_cell(cluster_positions, padding=vacuum_padding)

    extracted_atoms = Atoms(
        symbols=[atoms[index].symbol for index in selected],
        positions=cluster_positions,
        cell=cluster_cell,
        pbc=False,
    )

    local_distances = distances[selected]
    regions = assign_radial_regions(
        local_distances,
        core_radius=float(extraction_cfg.get("core_radius", 4.0)),
        extract_radius=extract_radius,
        boundary_thickness=float(extraction_cfg.get("boundary_thickness", 2.0)),
    )
    region_labels = region_label_vector(len(selected), regions)
    hotspot_local = [selected.index(index) for index in hotspot_indices if index in selected]

    return ExtractedRegion(
        atoms=extracted_atoms,
        original_indices=selected,
        core_indices=regions["core"],
        inner_buffer_indices=regions["inner_buffer"],
        outer_buffer_indices=regions["outer_buffer"],
        boundary_indices=regions["boundary"],
        h_cap_indices=[],
        hotspot_indices=hotspot_local,
        region_labels=region_labels,
        metadata={
            "mode": "cluster",
            "center": np.asarray(center, dtype=float).tolist(),
            "extract_radius": extract_radius,
            "core_radius": float(extraction_cfg.get("core_radius", 4.0)),
            "boundary_thickness": float(extraction_cfg.get("boundary_thickness", 2.0)),
            "vacuum_padding": vacuum_padding,
        },
    )
