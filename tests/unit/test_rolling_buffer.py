"""Boundary-condition tests for rolling event buffers."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

import hotspot_al.buffer.rolling_buffer as rolling_buffer_module
from hotspot_al.buffer.rolling_buffer import RollingBuffer
from hotspot_al.models import FrameData


def _frame(step: int) -> FrameData:
    return FrameData(atoms=Atoms("H", positions=[[float(step), 0.0, 0.0]]), step=step, forces=np.zeros((1, 3)))


def test_push_no_pending_returns_none() -> None:
    buffer = RollingBuffer(pre_trigger_frames=2, post_trigger_frames=1)

    assert buffer.push(_frame(0)) is None


def test_capture_event_insufficient_pre_frames() -> None:
    buffer = RollingBuffer(pre_trigger_frames=5, post_trigger_frames=0)
    buffer.push(_frame(1))
    trigger = _frame(2)
    buffer.push(trigger)

    event = buffer.capture_event(trigger, hotspot_atoms=[0], ood_scores=np.array([7.0]), trigger_reason=["force"])

    assert event is not None
    assert [frame.step for frame in event.pre_frames] == [1]


def test_consecutive_triggers_on_consecutive_frames() -> None:
    buffer = RollingBuffer(pre_trigger_frames=1, post_trigger_frames=0)
    first = _frame(1)
    second = _frame(2)
    buffer.push(first)
    first_event = buffer.capture_event(first, hotspot_atoms=[0], ood_scores=np.array([7.0]), trigger_reason=["force"])
    buffer.push(second)
    second_event = buffer.capture_event(second, hotspot_atoms=[0], ood_scores=np.array([8.0]), trigger_reason=["force"])

    assert first_event is not None
    assert second_event is not None
    assert first_event.step == 1
    assert second_event.step == 2


def test_flush_unfinished_event() -> None:
    buffer = RollingBuffer(pre_trigger_frames=1, post_trigger_frames=3)
    trigger = _frame(10)
    buffer.push(trigger)
    assert buffer.capture_event(trigger, hotspot_atoms=[0], ood_scores=np.array([7.0]), trigger_reason=["force"]) is None

    event = buffer.flush()

    assert event is not None
    assert event.step == 10
    assert event.post_frames == []


def test_maxlen_overflow_preserves_available_pre_frames() -> None:
    buffer = RollingBuffer(pre_trigger_frames=3, post_trigger_frames=0, maxlen=2)
    for step in range(4):
        buffer.push(_frame(step))
    trigger = _frame(4)
    buffer.push(trigger)

    event = buffer.capture_event(trigger, hotspot_atoms=[0], ood_scores=np.array([7.0]), trigger_reason=["force"])

    assert event is not None
    assert [frame.step for frame in event.pre_frames] == [3]


def test_event_metadata_propagation() -> None:
    buffer = RollingBuffer(pre_trigger_frames=1, post_trigger_frames=0)
    trigger = _frame(5)
    buffer.push(trigger)

    event = buffer.capture_event(
        trigger,
        hotspot_atoms=[0],
        ood_scores=np.array([7.0]),
        trigger_reason=["force"],
        event_id="evt-custom",
        backend="allegro",
        model_version="v1",
        metadata={"source": "unit"},
    )

    assert event is not None
    assert event.event_id == "evt-custom"
    assert event.backend == "allegro"
    assert event.model_version == "v1"
    assert event.metadata["source"] == "unit"


def test_finalize_pending_restores_pending_event_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    buffer = RollingBuffer(pre_trigger_frames=1, post_trigger_frames=0)
    trigger = _frame(8)
    buffer.push(trigger)

    def fail_event_record(**_kwargs):
        raise RuntimeError("event construction failed")

    monkeypatch.setattr(rolling_buffer_module, "EventRecord", fail_event_record)

    with pytest.raises(RuntimeError, match="event construction failed"):
        buffer.capture_event(trigger, hotspot_atoms=[0], ood_scores=np.array([7.0]), trigger_reason=["force"])

    assert buffer._pending is not None
    assert buffer._pending.trigger_frame.step == 8
