"""Tests for model dataclass serialization helpers."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from ase import Atoms

from hotspot_al.models import EventRecord, ExtractedRegion, FrameData, Hotspot


def _frame(step: int = 3) -> FrameData:
    atoms = Atoms("OH", positions=[[0.0, 0.0, 0.0], [0.0, 0.8, 0.0]], cell=np.diag([6.0, 6.0, 6.0]), pbc=True)
    atoms.arrays["charges"] = np.array([-0.8, 0.8])
    atoms.info["source"] = "unit"
    return FrameData(
        atoms=atoms,
        step=step,
        time=1.5,
        forces=np.ones((2, 3)),
        velocities=np.full((2, 3), 0.25),
        energy=-1.2,
        metadata={"frame": step},
    )


def test_frame_data_pickle_roundtrip_preserves_atoms_and_arrays() -> None:
    restored = pickle.loads(pickle.dumps(_frame()))

    assert restored.step == 3
    assert restored.atoms.get_chemical_symbols() == ["O", "H"]
    assert np.allclose(restored.atoms.positions, [[0.0, 0.0, 0.0], [0.0, 0.8, 0.0]])
    assert np.allclose(restored.atoms.arrays["charges"], [-0.8, 0.8])
    assert np.allclose(restored.forces, np.ones((2, 3)))
    assert restored.metadata == {"frame": 3}


def test_extracted_region_pickle_roundtrip_preserves_masks() -> None:
    region = ExtractedRegion(
        atoms=_frame().atoms,
        original_indices=[10, 11],
        core_indices=[0],
        inner_buffer_indices=[1],
        outer_buffer_indices=[],
        boundary_indices=[],
        h_cap_indices=[],
        hotspot_indices=[0],
        region_labels=["core", "inner_buffer"],
        mask_weights=np.array([1.0, 0.3]),
        metadata={"mode": "cluster"},
    )

    restored = pickle.loads(pickle.dumps(region))

    assert restored.original_indices == [10, 11]
    assert restored.region_labels == ["core", "inner_buffer"]
    assert np.allclose(restored.mask_weights, [1.0, 0.3])
    assert restored.metadata["mode"] == "cluster"


def test_event_record_pickle_roundtrip_preserves_nested_frames() -> None:
    event = EventRecord(
        pre_frames=[_frame(1)],
        trigger_frame=_frame(2),
        post_frames=[_frame(3)],
        hotspot_atoms=[0],
        ood_scores=np.array([7.0, 1.0]),
        trigger_reason=["force_large"],
        step=2,
        time=1.0,
        event_id="evt-2",
        backend="allegro",
        model_version="v1",
        metadata={"source": "unit"},
    )

    restored = pickle.loads(pickle.dumps(event))

    assert restored.event_id == "evt-2"
    assert restored.pre_frames[0].step == 1
    assert restored.trigger_frame.step == 2
    assert restored.post_frames[0].step == 3
    assert np.allclose(restored.ood_scores, [7.0, 1.0])


def test_hotspot_atom_indices_alias_warns() -> None:
    hotspot = Hotspot(
        core_atom_indices=[1, 2],
        center=np.zeros(3),
        max_score=9.0,
        trigger_reasons=["force_large"],
        step=5,
    )

    with pytest.warns(DeprecationWarning, match="core_atom_indices"):
        assert hotspot.atom_indices == [1, 2]
