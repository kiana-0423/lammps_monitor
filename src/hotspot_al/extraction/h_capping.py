"""Hydrogen capping for truncated covalent bonds in extracted clusters."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers

from hotspot_al.models import ExtractedRegion
from hotspot_al.utils.neighbor import bonded_neighbors
from hotspot_al.utils.periodic import mic_displacement


_METAL_LIKE = {
    "Li",
    "Na",
    "K",
    "Mg",
    "Ca",
    "Al",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Zr",
    "Mo",
    "Ag",
    "Sn",
    "Pt",
    "Au",
}
_OXIDE_FORMERS = _METAL_LIKE | {"Si"}


def add_h_caps(
    original_atoms: Atoms,
    region: ExtractedRegion,
    *,
    config: dict[str, Any],
) -> ExtractedRegion:
    """Add hydrogen caps along broken bonds at the region boundary.

    The implementation is conservative by default:
    - only atoms in boundary/outer buffer are considered,
    - core atoms are never capped unless explicitly allowed,
    - only elements with configured X-H bond lengths are capped.
    """

    h_cfg = config.get("h_capping", config)
    if not h_cfg.get("enabled", True):
        return region

    allow_core_capping = bool(h_cfg.get("allow_core_capping", False))
    covalent_scale = float(h_cfg.get("covalent_scale", 1.2))
    bond_lengths = {str(key): float(value) for key, value in h_cfg.get("bond_lengths", {}).items()}
    symbols = set(original_atoms.get_chemical_symbols())
    if bool(h_cfg.get("disabled_for_oxides_by_default", True)) and "O" in symbols and any(symbol in _OXIDE_FORMERS for symbol in symbols):
        return region

    adjacency = bonded_neighbors(
        original_atoms.get_positions(),
        original_atoms.get_atomic_numbers(),
        cell=original_atoms.cell.array,
        pbc=original_atoms.pbc,
        scale=covalent_scale,
    )

    selected_set = set(region.original_indices)
    cap_allowed = set(region.boundary_indices) | set(region.outer_buffer_indices)
    if allow_core_capping:
        cap_allowed |= set(region.inner_buffer_indices)

    updated_region = deepcopy(region)
    base_atoms = updated_region.atoms.copy()
    h_positions: list[np.ndarray] = []
    h_metadata: list[dict[str, Any]] = []
    parent_local_indices: list[int] = []

    for local_index, original_index in enumerate(region.original_indices):
        if local_index not in cap_allowed:
            continue
        if not allow_core_capping and local_index in region.core_indices:
            continue
        symbol = original_atoms[original_index].symbol
        if symbol not in bond_lengths:
            continue
        if bool(h_cfg.get("disabled_for_metals", True)) and symbol in _METAL_LIKE:
            continue
        for neighbor in adjacency.get(original_index, []):
            if neighbor in selected_set:
                continue
            direction = mic_displacement(
                original_atoms.positions[original_index],
                original_atoms.positions[neighbor],
                cell=original_atoms.cell.array,
                pbc=original_atoms.pbc,
            )
            norm = float(np.linalg.norm(direction))
            if norm < 1.0e-8:
                continue
            h_position = base_atoms.positions[local_index] + bond_lengths[symbol] * (direction / norm)
            h_positions.append(h_position)
            parent_local_indices.append(local_index)
            h_metadata.append(
                {
                    "parent_local_index": local_index,
                    "parent_original_index": original_index,
                    "broken_partner_original_index": neighbor,
                    "bond_length": bond_lengths[symbol],
                }
            )

    if not h_positions:
        return updated_region

    base_atoms.extend(Atoms(symbols=["H"] * len(h_positions), positions=h_positions))
    start_index = len(region.atoms)
    new_h_indices = list(range(start_index, start_index + len(h_positions)))
    updated_region.atoms = base_atoms
    updated_region.original_indices = [*region.original_indices, *([-1] * len(h_positions))]
    updated_region.h_cap_indices = [*region.h_cap_indices, *new_h_indices]
    updated_region.region_labels = [
        *(region.region_labels or ["unassigned"] * len(region.atoms)),
        *(["h_cap"] * len(h_positions)),
    ]
    if updated_region.mask_weights is not None:
        updated_region.mask_weights = np.concatenate([updated_region.mask_weights, np.zeros(len(h_positions), dtype=float)])
    updated_region.metadata = {
        **region.metadata,
        "h_caps": h_metadata,
        "h_cap_parent_local_indices": parent_local_indices,
    }
    return updated_region
