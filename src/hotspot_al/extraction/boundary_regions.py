"""Helpers for assigning extracted atoms to radial regions."""

from __future__ import annotations

import numpy as np


def assign_radial_regions(
    distances: np.ndarray,
    *,
    core_radius: float,
    extract_radius: float,
    boundary_thickness: float,
) -> dict[str, list[int]]:
    """Assign atom indices to core, inner buffer, outer shell, and boundary."""

    distances = np.asarray(distances, dtype=float)
    boundary_start = max(core_radius, extract_radius - boundary_thickness)
    core = np.where(distances <= core_radius)[0].tolist()
    inner_buffer = np.where((distances > core_radius) & (distances < boundary_start))[0].tolist()
    outer_buffer = np.where((distances >= boundary_start) & (distances <= extract_radius))[0].tolist()
    boundary = list(outer_buffer)
    return {
        "core": core,
        "inner_buffer": inner_buffer,
        "outer_buffer": outer_buffer,
        "boundary": boundary,
    }


def region_label_vector(n_atoms: int, regions: dict[str, list[int]]) -> list[str]:
    """Return a per-atom region label vector."""

    labels = ["unassigned"] * n_atoms
    for index in regions.get("outer_buffer", []):
        labels[index] = "outer_buffer"
    for index in regions.get("boundary", []):
        labels[index] = "boundary"
    for index in regions.get("inner_buffer", []):
        labels[index] = "inner_buffer"
    for index in regions.get("core", []):
        labels[index] = "core"
    return labels
