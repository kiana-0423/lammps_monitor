"""Slab patch extraction for surface and interface systems."""

from __future__ import annotations

from typing import Any

import numpy as np
from ase import Atoms

from hotspot_al.extraction.boundary_regions import assign_radial_regions, region_label_vector
from hotspot_al.models import ExtractedRegion
from hotspot_al.utils.geometry import distances_to_group, hotspot_center
from hotspot_al.utils.periodic import mic_displacements_from_reference


def extract_slab_patch(
    atoms: Atoms,
    hotspot_indices: list[int],
    *,
    config: dict[str, Any],
    xy_lengths: tuple[float, float] | None = None,
    z_thickness: float | None = None,
) -> ExtractedRegion:
    """Extract a periodic slab patch around a hotspot.

    This is a conservative baseline implementation: x-y periodicity is
    preserved while atoms outside a local lateral window are removed.
    """

    if not hotspot_indices:
        msg = "At least one hotspot atom is required for slab extraction."
        raise ValueError(msg)

    extraction_cfg = config.get("extraction", config)
    positions = atoms.get_positions()
    center = hotspot_center(positions, hotspot_indices, cell=atoms.cell.array, pbc=atoms.pbc)
    xy_lengths = xy_lengths or (
        2.0 * float(extraction_cfg.get("extract_radius", 10.0)),
        2.0 * float(extraction_cfg.get("extract_radius", 10.0)),
    )
    z_half = 0.5 * float(z_thickness or 2.0 * extraction_cfg.get("extract_radius", 10.0))
    displacements = mic_displacements_from_reference(center, positions, cell=atoms.cell.array, pbc=atoms.pbc)
    lateral_mask = (np.abs(displacements[:, 0]) <= xy_lengths[0] * 0.5) & (np.abs(displacements[:, 1]) <= xy_lengths[1] * 0.5)
    vertical_mask = np.abs(displacements[:, 2]) <= z_half
    selected = np.where(lateral_mask & vertical_mask)[0].tolist()
    if not selected:
        selected = sorted(hotspot_indices)

    local_positions = displacements[selected]
    mins = local_positions.min(axis=0)
    shifted = local_positions - mins
    cell = atoms.cell.array.copy()
    cell[0, 0] = max(xy_lengths[0], 1.0)
    cell[1, 1] = max(xy_lengths[1], 1.0)
    cell[2, 2] = max(float(shifted[:, 2].ptp() + 6.0), 6.0)

    extracted_atoms = Atoms(
        symbols=[atoms[index].symbol for index in selected],
        positions=shifted,
        cell=cell,
        pbc=[True, True, False],
    )

    distances = distances_to_group(positions, hotspot_indices, cell=atoms.cell.array, pbc=atoms.pbc)[selected]
    regions = assign_radial_regions(
        distances,
        core_radius=float(extraction_cfg.get("core_radius", 4.0)),
        extract_radius=float(extraction_cfg.get("extract_radius", 10.0)),
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
            "mode": "slab",
            "center": center.tolist(),
            "xy_lengths": list(xy_lengths),
            "z_thickness": 2.0 * z_half,
        },
    )
