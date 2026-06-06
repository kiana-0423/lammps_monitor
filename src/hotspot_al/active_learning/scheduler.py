"""Simple scheduling helpers for active-learning rounds."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from hotspot_al.models import EventRecord


@dataclass(slots=True)
class RoundSummary:
    """Bookkeeping summary for one active-learning round."""

    round_id: int
    n_events: int
    n_candidates: int
    n_selected: int


@dataclass(slots=True)
class ScheduledTask:
    """A lightweight task created from an online OOD event."""

    task_id: str
    event: EventRecord
    status: str = "pending"
    metadata: dict[str, Any] = field(default_factory=dict)


TaskSubmitter = Callable[[ScheduledTask], None]


class OnlineEventScheduler:
    """Queue online OOD events and optionally submit them to an external system."""

    def __init__(self, *, submitter: TaskSubmitter | None = None) -> None:
        self.submitter = submitter
        self.pending: deque[ScheduledTask] = deque()
        self.submitted: list[ScheduledTask] = []
        self.failed: list[ScheduledTask] = []

    def schedule_event(self, event: EventRecord, *, metadata: dict[str, Any] | None = None) -> ScheduledTask:
        """Create a task for an event and enqueue it."""

        task = ScheduledTask(
            task_id=event.event_id or f"event-{event.step}",
            event=event,
            metadata=dict(metadata or {}),
        )
        self.pending.append(task)
        return task

    def submit_next(self) -> ScheduledTask | None:
        """Submit one pending task if a submitter is configured."""

        if not self.pending:
            return None
        task = self.pending.popleft()
        if self.submitter is None:
            task.status = "queued"
            self.submitted.append(task)
            return task
        try:
            self.submitter(task)
        except Exception as exc:
            task.status = "failed"
            task.metadata["error"] = str(exc)
            self.failed.append(task)
            return task
        task.status = "submitted"
        self.submitted.append(task)
        return task

    def drain(self) -> list[ScheduledTask]:
        """Submit or mark queued every pending task."""

        drained: list[ScheduledTask] = []
        while self.pending:
            task = self.submit_next()
            if task is not None:
                drained.append(task)
        return drained


def summarize_round(round_id: int, *, n_events: int, n_candidates: int, n_selected: int) -> RoundSummary:
    """Build a small round summary object."""

    return RoundSummary(round_id=round_id, n_events=n_events, n_candidates=n_candidates, n_selected=n_selected)
