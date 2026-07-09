"""Tests for block-aware PHAL extraction."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from hotspot_al.config import load_config
from hotspot_al.cp2k.cp2k_input import write_cp2k_inputs
from hotspot_al.extraction.block import (
    BlockCooldownTracker,
    anomalous_blocks,
    assign_atoms_to_spatial_blocks,
    extract_block_region,
    extract_block_regions,
    merge_adjacent_blocks,
)
from hotspot_al.extraction.cluster_extractor import extract_cluster_region
from hotspot_al.models import ExtractedRegion
from hotspot_al.training.mask_generator import generate_atom_mask


def _block_config() -> dict:
    return {
        "extraction": {
            "mode": "block",
            "block": {
                "scheme": "spatial_grid",
                "size": [5.0, 5.0, 5.0],
                "halo": 8.0,
                "merge_adjacent": True,
                "max_merged_blocks": 4,
                "cooldown_steps": 10,
                "label_region": {"type": "block_core", "shrink": 0.0},
                "buffer": {"inner": 4.1, "outer": 8.0},
                "frozen": {"enabled": True, "thickness": 1.5},
                "max_atoms": 20,
                "min_atoms": 1,
            },
        },
        "training_mask": {
            "label_core": 1.0,
            "core": 1.0,
            "inner_buffer": 0.2,
            "outer_buffer": 0.1,
            "frozen_boundary": 0.0,
            "boundary": 0.0,
            "h_cap": 0.0,
        },
    }


def _grid_atoms() -> Atoms:
    return Atoms(
        "H4",
        positions=[
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [6.0, 1.0, 1.0],
            [9.5, 9.5, 9.5],
        ],
        cell=np.diag([10.0, 10.0, 10.0]),
        pbc=True,
    )


def _role_atoms() -> Atoms:
    return Atoms(
        "H6",
        positions=[
            [1.0, 1.0, 1.0],
            [2.0, 1.0, 1.0],
            [6.0, 1.0, 1.0],
            [7.5, 1.0, 1.0],
            [9.0, 1.0, 1.0],
            [15.0, 1.0, 1.0],
        ],
        cell=np.diag([20.0, 10.0, 10.0]),
        pbc=False,
    )


def test_assign_atoms_to_spatial_blocks() -> None:
    atom_to_block = assign_atoms_to_spatial_blocks(_grid_atoms(), [5.0, 5.0, 5.0])

    assert atom_to_block[0] == (0, 0, 0)
    assert atom_to_block[1] == (0, 0, 0)
    assert atom_to_block[2] == (1, 0, 0)
    assert atom_to_block[3] == (1, 1, 1)


def test_ood_atoms_map_to_unique_blocks() -> None:
    atoms = _grid_atoms()
    atom_to_block = assign_atoms_to_spatial_blocks(atoms, [5.0, 5.0, 5.0])

    assert anomalous_blocks(atom_to_block, [0, 1]) == [(0, 0, 0)]

    config = _block_config()
    config["extraction"]["block"]["merge_adjacent"] = False
    regions = extract_block_regions(atoms, [0, 1], config=config, step=1)
    assert len(regions) == 1
    assert regions[0].metadata["block_ids"] == [[0, 0, 0]]


def test_merge_adjacent_blocks() -> None:
    groups = merge_adjacent_blocks([(0, 0, 0), (1, 0, 0), (3, 0, 0)])

    assert groups == [[(0, 0, 0), (1, 0, 0)], [(3, 0, 0)]]


def test_merge_adjacent_blocks_wraps_across_pbc_boundary() -> None:
    groups = merge_adjacent_blocks(
        [(0, 0, 0), (3, 0, 0)],
        grid_shape=(4, 4, 4),
        pbc=(True, True, True),
    )

    assert groups == [[(0, 0, 0), (3, 0, 0)]]


def test_merge_adjacent_blocks_no_wrap_without_pbc() -> None:
    groups = merge_adjacent_blocks(
        [(0, 0, 0), (3, 0, 0)],
        grid_shape=(4, 4, 4),
        pbc=(False, False, False),
    )

    assert groups == [[(0, 0, 0)], [(3, 0, 0)]]


def test_merge_adjacent_blocks_2d_periodic() -> None:
    groups = merge_adjacent_blocks(
        [(0, 0, 0), (2, 2, 0)],
        grid_shape=(3, 3, 3),
        pbc=(True, True, False),
    )

    assert groups == [[(0, 0, 0), (2, 2, 0)]]


def test_cooldown_tracker() -> None:
    tracker = BlockCooldownTracker(cooldown_steps=10)
    block_ids = [(0, 0, 0)]

    assert not tracker.should_skip(block_ids, 5)
    tracker.update(block_ids, 5)
    assert tracker.should_skip(block_ids, 12)
    assert not tracker.should_skip(block_ids, 15)


def test_cooldown_tracker_prunes_expired_entries() -> None:
    tracker = BlockCooldownTracker(cooldown_steps=10)
    for index in range(10001):
        tracker.update([(index, 0, 0)], step=index)
    tracker.update([(0, 0, 0)], step=20000)

    assert all(step >= 19980 for step in tracker.last_labeled_step.values())


def test_extract_block_region_roles_are_disjoint_and_valid() -> None:
    region = extract_block_region(_role_atoms(), [(0, 0, 0)], ood_atom_indices=[0], config=_block_config(), step=4)

    groups = [region.core_indices, region.inner_buffer_indices, region.outer_buffer_indices, region.boundary_indices]
    flattened = [index for group in groups for index in group]
    assert len(flattened) == len(set(flattened))
    assert all(0 <= index < len(region.atoms) for index in flattened)
    assert set(region.metadata["atom_role"]) >= {"label_core", "inner_buffer", "outer_buffer", "frozen_boundary"}
    assert region.metadata["extraction_mode"] == "block"
    assert region.h_cap_indices == []
    assert region.metadata["inner_buffer_definition"] == "distance_to_core_atoms"
    assert region.metadata["outer_buffer_definition"] == "distance_to_core_atoms"


def test_training_mask_weights_for_block_roles() -> None:
    region = extract_block_region(_role_atoms(), [(0, 0, 0)], ood_atom_indices=[0], config=_block_config(), step=4)

    mask = generate_atom_mask(region, _block_config())

    for index, role in enumerate(region.metadata["atom_role"]):
        if role == "label_core":
            assert mask[index] == pytest.approx(1.0)
        elif role == "inner_buffer":
            assert mask[index] == pytest.approx(0.2)
        elif role == "outer_buffer":
            assert mask[index] == pytest.approx(0.1)
        elif role == "frozen_boundary":
            assert mask[index] == pytest.approx(0.0)


def test_max_atoms_behavior_trims_buffers_and_raises_for_core_overflow() -> None:
    config = _block_config()
    config["extraction"]["block"]["max_atoms"] = 2
    region = extract_block_region(_role_atoms(), [(0, 0, 0)], ood_atom_indices=[0], config=config, step=4)

    assert len(region.atoms) == 2
    assert sorted(region.core_indices) == [0, 1]

    config["extraction"]["block"]["max_atoms"] = 1
    with pytest.raises(ValueError, match="max_atoms_exceeded"):
        extract_block_region(_role_atoms(), [(0, 0, 0)], ood_atom_indices=[0], config=config, step=4)


def test_extract_block_region_pbc_centroid_correct() -> None:
    atoms = Atoms(
        "H4",
        positions=[[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [9.5, 0.5, 0.5], [8.5, 0.5, 0.5]],
        cell=np.diag([10.0, 10.0, 10.0]),
        pbc=True,
    )
    config = _block_config()
    config["extraction"]["block"]["size"] = [5.0, 5.0, 5.0]
    config["extraction"]["block"]["buffer"] = {"inner": 2.0, "outer": 4.0}
    config["extraction"]["block"]["halo"] = 4.0

    region = extract_block_region(atoms, [(0, 0, 0)], ood_atom_indices=[0], config=config, step=1)

    positions = region.atoms.get_positions()
    max_dist = np.max(np.linalg.norm(positions[:, None] - positions[None, :], axis=-1))
    assert max_dist < 8.0


def test_backward_compatibility_cluster_extraction_still_works() -> None:
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], pbc=False)
    config = {
        "extraction": {
            "mode": "cluster",
            "core_radius": 1.1,
            "extract_radius": 2.0,
            "boundary_thickness": 0.5,
            "min_atoms": 1,
            "max_atoms": 10,
            "vacuum_padding": 2.0,
        }
    }

    region = extract_cluster_region(atoms, [0], config=config)

    assert region.core_indices == [0, 1]


def test_cp2k_input_writes_frozen_boundary_constraints(tmp_path) -> None:
    region = ExtractedRegion(
        atoms=Atoms("H3", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], cell=np.diag([8.0, 8.0, 8.0])),
        original_indices=[0, 1, 2],
        core_indices=[0],
        inner_buffer_indices=[],
        outer_buffer_indices=[],
        boundary_indices=[1, 2],
        h_cap_indices=[],
        hotspot_indices=[0],
        region_labels=["label_core", "frozen_boundary", "frozen_boundary"],
        metadata={"extraction_mode": "block"},
    )

    written = write_cp2k_inputs(region, tmp_path, config=load_config(), job_name="block")

    assert "hopt_input" in written
    text = written["hopt_input"].read_text(encoding="utf-8")
    assert "&FIXED_ATOMS" in text
    assert "LIST 2 3" in text
    assert "single_point_input" in written
