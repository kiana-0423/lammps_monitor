"""Block-aware PHAL extraction for stable local relabeling regions."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from ase import Atoms

from hotspot_al.models import ExtractedRegion
from hotspot_al.utils.geometry import distances_to_group, hotspot_center, padded_cluster_cell
from hotspot_al.utils.periodic import mic_displacements_from_reference

BlockId = tuple[int, int, int]


@dataclass(slots=True)
class BlockCooldownTracker:
    """Track recently labeled blocks to avoid repeated DFT submissions."""

    cooldown_steps: int
    last_labeled_step: dict[BlockId, int] | None = None

    def __post_init__(self) -> None:
        if self.last_labeled_step is None:
            self.last_labeled_step = {}

    def should_skip(self, block_ids: Iterable[BlockId], step: int) -> bool:
        """Return True when any block is still inside the cooldown window."""

        if self.cooldown_steps <= 0:
            return False
        assert self.last_labeled_step is not None
        for block_id in block_ids:
            last_step = self.last_labeled_step.get(block_id)
            if last_step is not None and step - last_step < self.cooldown_steps:
                return True
        return False

    def update(self, block_ids: Iterable[BlockId], step: int) -> None:
        """Record that ``block_ids`` were scheduled for labeling at ``step``."""

        assert self.last_labeled_step is not None
        for block_id in block_ids:
            self.last_labeled_step[block_id] = step
        should_prune = len(self.last_labeled_step) > 10000 or (
            self.cooldown_steps > 0 and step % self.cooldown_steps == 0
        )
        if should_prune:
            expired = [
                block_id
                for block_id, last_step in self.last_labeled_step.items()
                if step - last_step > self.cooldown_steps * 2
            ]
            for block_id in expired:
                del self.last_labeled_step[block_id]


@dataclass(slots=True)
class _BlockAtomSelection:
    core_original: list[int]
    inner_original: list[int]
    outer_original: list[int]
    frozen_original: list[int]
    distances: np.ndarray
    inner_radius: float
    outer_radius: float
    frozen_thickness: float


def block_grid_shape(atoms: Atoms, block_size: Iterable[float]) -> tuple[int, int, int]:
    """Return the number of spatial-grid blocks along each cell direction."""

    size = np.asarray(list(block_size), dtype=float)
    if size.shape != (3,) or np.any(size <= 0.0):
        msg = "extraction.block.size must contain three positive numbers."
        raise ValueError(msg)
    lengths = np.asarray(atoms.cell.lengths(), dtype=float)
    lengths = np.where(lengths > 1.0e-12, lengths, size)
    raw_shape = np.maximum(1, np.ceil(lengths / size)).astype(int).tolist()
    return (int(raw_shape[0]), int(raw_shape[1]), int(raw_shape[2]))


def assign_atoms_to_spatial_blocks(
    atoms: Atoms,
    block_size: Iterable[float],
    *,
    pbc: bool | tuple[bool, bool, bool] | np.ndarray | None = None,
) -> dict[int, BlockId]:
    """Assign atoms to stable spatial-grid block ids.

    Scaled coordinates are used so non-orthogonal cells are handled by a regular
    grid in fractional space. For periodic axes, coordinates are wrapped into
    ``[0, 1)`` before block assignment.
    """

    grid_shape = block_grid_shape(atoms, block_size)
    pbc_flags = np.asarray(atoms.pbc if pbc is None else pbc, dtype=bool)
    if pbc_flags.shape == ():
        pbc_flags = np.repeat(bool(pbc_flags), 3)
    scaled = np.asarray(atoms.get_scaled_positions(wrap=False), dtype=float)
    for axis, periodic in enumerate(pbc_flags):
        if periodic:
            scaled[:, axis] = np.mod(scaled[:, axis], 1.0)
    block_coords = np.floor(scaled * np.asarray(grid_shape, dtype=float)).astype(int)
    for axis, n_blocks in enumerate(grid_shape):
        block_coords[:, axis] = np.clip(block_coords[:, axis], 0, n_blocks - 1)
    return {
        index: (int(block_coords[index][0]), int(block_coords[index][1]), int(block_coords[index][2]))
        for index in range(len(atoms))
    }


def invert_block_mapping(atom_to_block: dict[int, BlockId]) -> dict[BlockId, list[int]]:
    """Return block id to atom-index mapping."""

    block_to_atoms: dict[BlockId, list[int]] = defaultdict(list)
    for atom_index, block_id in atom_to_block.items():
        block_to_atoms[block_id].append(atom_index)
    return {block_id: sorted(indices) for block_id, indices in block_to_atoms.items()}


def anomalous_blocks(atom_to_block: dict[int, BlockId], atom_indices: Iterable[int]) -> list[BlockId]:
    """Map anomalous atom indices to unique sorted block ids."""

    return sorted({atom_to_block[index] for index in atom_indices if index in atom_to_block})


def merge_adjacent_blocks(
    block_ids: Iterable[BlockId],
    *,
    max_merged_blocks: int | None = None,
    grid_shape: tuple[int, int, int] | None = None,
    pbc: tuple[bool, bool, bool] = (False, False, False),
) -> list[list[BlockId]]:
    """Merge 26-neighbor-connected block ids into block groups."""

    remaining = set(block_ids)
    groups: list[list[BlockId]] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group = [start]
        queue: deque[BlockId] = deque([start])
        while queue:
            current = queue.popleft()
            connected = [
                other
                for other in remaining
                if _blocks_are_adjacent(current, other, grid_shape=grid_shape, pbc=pbc)
            ]
            for other in connected:
                remaining.remove(other)
                group.append(other)
                queue.append(other)
        group = sorted(group)
        if max_merged_blocks is not None and max_merged_blocks > 0 and len(group) > max_merged_blocks:
            groups.extend(
                _split_connected_block_group(
                    group,
                    max_blocks=max_merged_blocks,
                    grid_shape=grid_shape,
                    pbc=pbc,
                )
            )
        else:
            groups.append(group)
    return groups


def extract_block_regions(
    atoms: Atoms,
    ood_atom_indices: list[int],
    *,
    config: dict[str, Any],
    step: int | None = None,
    cooldown_tracker: BlockCooldownTracker | None = None,
) -> list[ExtractedRegion]:
    """Build block-localized regions for anomalous atoms."""

    if not ood_atom_indices:
        return []
    block_cfg = _block_config(config)
    block_size = _block_size(block_cfg)
    atom_to_block = assign_atoms_to_spatial_blocks(atoms, block_size)
    block_ids = anomalous_blocks(atom_to_block, ood_atom_indices)
    if not block_ids:
        return []
    if bool(block_cfg.get("merge_adjacent", True)):
        raw_pbc = np.asarray(atoms.pbc, dtype=bool).tolist()
        pbc_flags = (bool(raw_pbc[0]), bool(raw_pbc[1]), bool(raw_pbc[2]))
        block_groups = merge_adjacent_blocks(
            block_ids,
            max_merged_blocks=int(block_cfg.get("max_merged_blocks", 4)),
            grid_shape=block_grid_shape(atoms, block_size),
            pbc=pbc_flags,
        )
    else:
        block_groups = [[block_id] for block_id in block_ids]

    regions: list[ExtractedRegion] = []
    current_step = int(step or 0)
    for block_group in block_groups:
        if cooldown_tracker is not None and cooldown_tracker.should_skip(block_group, current_step):
            continue
        region = extract_block_region(
            atoms,
            block_group,
            ood_atom_indices=ood_atom_indices,
            atom_to_block=atom_to_block,
            config=config,
            step=step,
        )
        regions.append(region)
        if cooldown_tracker is not None:
            cooldown_tracker.update(block_group, current_step)
    return regions


def extract_block_region(
    atoms: Atoms,
    block_ids: list[BlockId],
    *,
    ood_atom_indices: list[int],
    config: dict[str, Any],
    step: int | None = None,
    atom_to_block: dict[int, BlockId] | None = None,
) -> ExtractedRegion:
    """Extract one block core plus halo/buffer/frozen-boundary region."""

    block_cfg = _block_config(config)
    atom_to_block = atom_to_block or assign_atoms_to_spatial_blocks(atoms, _block_size(block_cfg))
    selected_blocks = set(block_ids)
    max_atoms = int(block_cfg.get("max_atoms", config.get("extraction", {}).get("max_atoms", len(atoms))))
    min_atoms = int(block_cfg.get("min_atoms", config.get("extraction", {}).get("min_atoms", 1)))
    selection = _select_block_atoms(atoms, selected_blocks, atom_to_block=atom_to_block, block_cfg=block_cfg)
    selection.core_original, selection.inner_original, selection.outer_original, selection.frozen_original = _enforce_max_atoms(
        selection.core_original,
        selection.inner_original,
        selection.outer_original,
        selection.frozen_original,
        distances=selection.distances,
        max_atoms=max_atoms,
    )
    return _build_block_region(
        atoms,
        selection=selection,
        selected_blocks=selected_blocks,
        ood_atom_indices=ood_atom_indices,
        block_cfg=block_cfg,
        config=config,
        step=step,
        min_atoms=min_atoms,
    )


def _select_block_atoms(
    atoms: Atoms,
    selected_blocks: set[BlockId],
    *,
    atom_to_block: dict[int, BlockId],
    block_cfg: dict[str, Any],
) -> _BlockAtomSelection:
    """Classify original atom indices into block core, buffers, and frozen shell."""

    core_original = sorted(index for index, block_id in atom_to_block.items() if block_id in selected_blocks)
    if not core_original:
        msg = f"block extraction found no atoms in block group {sorted(selected_blocks)!r}."
        raise ValueError(msg)

    buffer_cfg = block_cfg.get("buffer", {})
    inner_radius = float(buffer_cfg.get("inner", 3.0))
    outer_radius = max(float(buffer_cfg.get("outer", 5.0)), float(block_cfg.get("halo", 5.0)), inner_radius)
    frozen_cfg = block_cfg.get("frozen", {})
    frozen_enabled = bool(frozen_cfg.get("enabled", True))
    frozen_thickness = max(0.0, float(frozen_cfg.get("thickness", 2.0))) if frozen_enabled else 0.0

    distances = distances_to_group(atoms.get_positions(), core_original, cell=atoms.cell.array, pbc=atoms.pbc)
    core_set = set(core_original)
    inner_original = sorted(
        index for index, distance in enumerate(distances) if index not in core_set and distance <= inner_radius
    )
    inner_set = set(inner_original)
    shell_original = [
        index
        for index, distance in enumerate(distances)
        if index not in core_set and index not in inner_set and distance <= outer_radius
    ]
    frozen_start = max(inner_radius, outer_radius - frozen_thickness)
    frozen_original = sorted(index for index in shell_original if frozen_enabled and distances[index] >= frozen_start)
    frozen_set = set(frozen_original)
    outer_original = sorted(index for index in shell_original if index not in frozen_set)
    return _BlockAtomSelection(
        core_original=core_original,
        inner_original=inner_original,
        outer_original=outer_original,
        frozen_original=frozen_original,
        distances=distances,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        frozen_thickness=frozen_thickness,
    )


def _build_block_region(
    atoms: Atoms,
    *,
    selection: _BlockAtomSelection,
    selected_blocks: set[BlockId],
    ood_atom_indices: list[int],
    block_cfg: dict[str, Any],
    config: dict[str, Any],
    step: int | None,
    min_atoms: int,
) -> ExtractedRegion:
    """Assemble an ``ExtractedRegion`` and block metadata from selected atoms."""

    selected = sorted(
        [
            *selection.core_original,
            *selection.inner_original,
            *selection.outer_original,
            *selection.frozen_original,
        ]
    )
    below_min_atoms = len(selected) < min_atoms
    extracted_atoms = _localized_atoms(atoms, selected)
    local_index = {original: index for index, original in enumerate(selected)}
    core_indices = [local_index[index] for index in selection.core_original]
    inner_indices = [local_index[index] for index in selection.inner_original]
    outer_indices = [local_index[index] for index in selection.outer_original]
    frozen_indices = [local_index[index] for index in selection.frozen_original]
    hotspot_indices = [local_index[index] for index in ood_atom_indices if index in local_index]
    atom_roles = _atom_roles(len(selected), core_indices, inner_indices, outer_indices, frozen_indices)
    region_labels = ["label_core" if role == "label_core" else role for role in atom_roles]

    return ExtractedRegion(
        atoms=extracted_atoms,
        original_indices=selected,
        core_indices=core_indices,
        inner_buffer_indices=inner_indices,
        outer_buffer_indices=outer_indices,
        boundary_indices=frozen_indices,
        h_cap_indices=[],
        hotspot_indices=hotspot_indices,
        region_labels=region_labels,
        metadata={
            "extraction_mode": "block",
            "mode": "block",
            "block_scheme": str(block_cfg.get("scheme", "spatial_grid")),
            "block_ids": [list(block_id) for block_id in sorted(selected_blocks)],
            "source_frame_id": step,
            "ood_atom_indices": list(ood_atom_indices),
            "training_weights": _training_weights(config),
            "original_indices": selected,
            "periodic": bool(np.any(atoms.pbc)),
            "cell": atoms.cell.array.tolist(),
            "block_size": list(_block_size(block_cfg)),
            "cooldown_steps": int(block_cfg.get("cooldown_steps", 0)),
            "inner_buffer": selection.inner_radius,
            "outer_buffer": selection.outer_radius,
            "frozen_thickness": selection.frozen_thickness,
            "inner_buffer_definition": "distance_to_core_atoms",
            "outer_buffer_definition": "distance_to_core_atoms",
            "core_max_distance_to_group": (
                float(max(selection.distances[index] for index in selection.core_original))
                if selection.core_original
                else 0.0
            ),
            "atom_role": atom_roles,
            "h_cap_enabled": False,
            "approximate_spatial_grid": True,
            "below_min_atoms": below_min_atoms,
        },
    )


def _block_config(config: dict[str, Any]) -> dict[str, Any]:
    extraction_cfg = config.get("extraction", {})
    if "block" not in extraction_cfg:
        msg = "extraction.mode='block' requires an extraction.block configuration."
        raise ValueError(msg)
    block_cfg = extraction_cfg["block"]
    if str(block_cfg.get("scheme", "spatial_grid")) != "spatial_grid":
        msg = "Only extraction.block.scheme='spatial_grid' is implemented."
        raise ValueError(msg)
    return block_cfg


def _block_size(block_cfg: dict[str, Any]) -> tuple[float, float, float]:
    raw_size = block_cfg.get("size")
    if raw_size is None:
        msg = "extraction.block.size is required for block extraction."
        raise ValueError(msg)
    size = tuple(float(value) for value in raw_size)
    if len(size) != 3 or any(value <= 0.0 for value in size):
        msg = "extraction.block.size must contain three positive numbers."
        raise ValueError(msg)
    return size


def _blocks_are_adjacent(
    left: BlockId,
    right: BlockId,
    *,
    grid_shape: tuple[int, int, int] | None = None,
    pbc: tuple[bool, bool, bool] = (False, False, False),
) -> bool:
    for axis in range(3):
        diff = abs(left[axis] - right[axis])
        if grid_shape is not None and pbc[axis] and grid_shape[axis] > 1:
            diff = min(diff, grid_shape[axis] - diff)
        if diff > 1:
            return False
    return True


def _split_connected_block_group(
    group: list[BlockId],
    *,
    max_blocks: int,
    grid_shape: tuple[int, int, int] | None,
    pbc: tuple[bool, bool, bool],
) -> list[list[BlockId]]:
    """Split an oversized connected group into bounded connected subgroups."""

    remaining = set(group)
    chunks: list[list[BlockId]] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        chunk = [start]
        queue: deque[BlockId] = deque([start])
        while queue and len(chunk) < max_blocks:
            current = queue.popleft()
            neighbors = sorted(
                other
                for other in remaining
                if _blocks_are_adjacent(current, other, grid_shape=grid_shape, pbc=pbc)
            )
            for other in neighbors:
                if len(chunk) >= max_blocks:
                    break
                remaining.remove(other)
                chunk.append(other)
                queue.append(other)
        chunks.append(sorted(chunk))
    return chunks


def _enforce_max_atoms(
    core: list[int],
    inner: list[int],
    outer: list[int],
    frozen: list[int],
    *,
    distances: np.ndarray,
    max_atoms: int,
) -> tuple[list[int], list[int], list[int], list[int]]:
    if len(core) > max_atoms:
        msg = f"max_atoms_exceeded: label_core has {len(core)} atoms but max_atoms={max_atoms}."
        raise ValueError(msg)
    inner = list(inner)
    outer = list(outer)
    frozen = list(frozen)
    frozen = _trim_group(frozen, target=max_atoms - len(core) - len(inner) - len(outer), distances=distances)
    outer = _trim_group(outer, target=max_atoms - len(core) - len(inner) - len(frozen), distances=distances)
    inner = _trim_group(inner, target=max_atoms - len(core) - len(outer) - len(frozen), distances=distances)
    total = len(core) + len(inner) + len(outer) + len(frozen)
    if total > max_atoms:
        msg = f"max_atoms_exceeded: could not reduce block region to max_atoms={max_atoms}."
        raise ValueError(msg)
    return sorted(core), sorted(inner), sorted(outer), sorted(frozen)


def _trim_group(indices: list[int], *, target: int, distances: np.ndarray) -> list[int]:
    if target >= len(indices):
        return indices
    if target <= 0:
        return []
    return sorted(sorted(indices, key=lambda index: float(distances[index]))[:target])


def _localized_atoms(atoms: Atoms, selected: list[int]) -> Atoms:
    positions = atoms.get_positions()
    center = hotspot_center(positions, selected, cell=atoms.cell.array, pbc=atoms.pbc)
    local_positions = mic_displacements_from_reference(center, positions[selected], cell=atoms.cell.array, pbc=atoms.pbc)
    mins = local_positions.min(axis=0)
    shifted = local_positions - mins + 2.0
    return Atoms(
        symbols=[atoms[index].symbol for index in selected],
        positions=shifted,
        cell=padded_cluster_cell(shifted, padding=4.0),
        pbc=False,
    )


def _atom_roles(
    n_atoms: int,
    core_indices: list[int],
    inner_indices: list[int],
    outer_indices: list[int],
    frozen_indices: list[int],
) -> list[str]:
    roles = ["unassigned"] * n_atoms
    for index in outer_indices:
        roles[index] = "outer_buffer"
    for index in frozen_indices:
        roles[index] = "frozen_boundary"
    for index in inner_indices:
        roles[index] = "inner_buffer"
    for index in core_indices:
        roles[index] = "label_core"
    return roles


def _training_weights(config: dict[str, Any]) -> dict[str, float]:
    mask_cfg = config.get("training_mask", {})
    return {
        "label_core": float(mask_cfg.get("label_core", mask_cfg.get("core", 1.0))),
        "inner_buffer": float(mask_cfg.get("inner_buffer", 0.2)),
        "outer_buffer": float(mask_cfg.get("outer_buffer", 0.0)),
        "frozen_boundary": float(mask_cfg.get("frozen_boundary", mask_cfg.get("boundary", 0.0))),
        "h_cap": float(mask_cfg.get("h_cap", 0.0)),
    }
