"""Tests for online event scheduling helpers."""

from __future__ import annotations

import numpy as np
from ase import Atoms

from hotspot_al.active_learning.scheduler import OnlineEventScheduler, ScheduledTask
from hotspot_al.models import EventRecord, FrameData


def _event() -> EventRecord:
    frame = FrameData(atoms=Atoms("H", positions=[[0.0, 0.0, 0.0]]), step=7, forces=np.zeros((1, 3)))
    return EventRecord(
        pre_frames=[],
        trigger_frame=frame,
        post_frames=[],
        hotspot_atoms=[0],
        ood_scores=np.array([3.0]),
        trigger_reason=["model_drift"],
        step=7,
        time=3.5,
        event_id="evt-test",
    )


def test_online_event_scheduler_submits_pending_tasks() -> None:
    submitted: list[str] = []

    def submitter(task: ScheduledTask) -> None:
        submitted.append(task.task_id)

    scheduler = OnlineEventScheduler(submitter=submitter)
    task = scheduler.schedule_event(_event())
    drained = scheduler.drain()

    assert task.task_id == "evt-test"
    assert [item.task_id for item in drained] == ["evt-test"]
    assert submitted == ["evt-test"]
    assert scheduler.submitted[0].status == "submitted"
