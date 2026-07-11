"""Rolling frame buffer for event-triggered hotspot capture."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import numpy as np

from hotspot_al.models import EventRecord, FrameData


@dataclass(slots=True)
class _PendingEvent:
    pre_frames: list[FrameData]
    trigger_frame: FrameData
    post_frames: list[FrameData]
    hotspot_atoms: list[int]
    ood_scores: np.ndarray
    trigger_reason: list[str]
    metadata: dict[str, Any]


class RollingBuffer:
    """Store recent frames and capture pre/post trigger context."""

    def __init__(self, pre_trigger_frames: int = 10, post_trigger_frames: int = 5, maxlen: int | None = None) -> None:
        self.pre_trigger_frames = pre_trigger_frames
        self.post_trigger_frames = post_trigger_frames
        self._frames: deque[FrameData] = deque(maxlen=maxlen or pre_trigger_frames + post_trigger_frames + 8)
        self._pending: _PendingEvent | None = None
        self.events: list[EventRecord] = []

    def push(self, frame: FrameData) -> EventRecord | None:
        """Append a frame and, if needed, finish a pending event."""

        self._frames.append(frame)
        if self._pending is None:
            return None
        if frame.step == self._pending.trigger_frame.step:
            return None
        self._pending.post_frames.append(frame)
        if len(self._pending.post_frames) >= self.post_trigger_frames:
            return self._finalize_pending()
        return None

    def capture_event(
        self,
        trigger_frame: FrameData,
        *,
        hotspot_atoms: list[int],
        ood_scores: np.ndarray,
        trigger_reason: list[str],
        event_id: str | None = None,
        backend: str | None = None,
        model_version: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EventRecord | None:
        """Start an event, or merge a repeated trigger into the pending event.

        Repeated triggers do not reset the post-trigger window.  The original
        trigger frame and all buffered frames are retained; hotspot atoms and
        reasons are unioned, while per-atom scores retain their maximum value.
        """

        if self._pending is not None:
            self._merge_trigger(
                trigger_frame,
                hotspot_atoms=hotspot_atoms,
                ood_scores=ood_scores,
                trigger_reason=trigger_reason,
                metadata=metadata,
            )
            return None

        frames = list(self._frames)
        if frames and frames[-1].step == trigger_frame.step:
            pre_frames = frames[:-1][-self.pre_trigger_frames :]
        else:
            pre_frames = frames[-self.pre_trigger_frames :]
        initial_metadata = dict(metadata or {})
        self._pending = _PendingEvent(
            pre_frames=pre_frames,
            trigger_frame=trigger_frame,
            post_frames=[],
            hotspot_atoms=list(hotspot_atoms),
            ood_scores=np.asarray(ood_scores, dtype=float).copy(),
            trigger_reason=list(trigger_reason),
            metadata={
                "event_id": event_id or f"evt-{trigger_frame.step}-{uuid4().hex[:8]}",
                "backend": backend,
                "model_version": model_version,
                **initial_metadata,
                "latest_trigger_step": trigger_frame.step,
                "trigger_steps": [trigger_frame.step],
                "trigger_count": 1,
                "trigger_metadata": [{"step": trigger_frame.step, **initial_metadata}],
            },
        )
        if self.post_trigger_frames == 0:
            return self._finalize_pending()
        return None

    def _merge_trigger(
        self,
        trigger_frame: FrameData,
        *,
        hotspot_atoms: list[int],
        ood_scores: np.ndarray,
        trigger_reason: list[str],
        metadata: dict[str, Any] | None,
    ) -> None:
        """Merge one trigger into the active event without extending it."""

        assert self._pending is not None
        pending = self._pending
        scores = np.asarray(ood_scores, dtype=float)
        if scores.shape != pending.ood_scores.shape:
            msg = (
                "Cannot merge trigger scores with a different shape: "
                f"{scores.shape} != {pending.ood_scores.shape}."
            )
            raise ValueError(msg)
        pending.hotspot_atoms = sorted(set(pending.hotspot_atoms).union(hotspot_atoms))
        pending.ood_scores = np.maximum(pending.ood_scores, scores)
        pending.trigger_reason = list(dict.fromkeys([*pending.trigger_reason, *trigger_reason]))
        pending.metadata["latest_trigger_step"] = trigger_frame.step
        pending.metadata["trigger_steps"] = list(
            dict.fromkeys([*pending.metadata.get("trigger_steps", [pending.trigger_frame.step]), trigger_frame.step])
        )
        pending.metadata["trigger_count"] = int(pending.metadata.get("trigger_count", 1)) + 1
        pending.metadata.setdefault("trigger_metadata", []).append({"step": trigger_frame.step, **dict(metadata or {})})

    def flush(self) -> EventRecord | None:
        """Finalize the pending event even if not enough post frames were observed."""

        if self._pending is None:
            return None
        return self._finalize_pending()

    def _finalize_pending(self) -> EventRecord:
        assert self._pending is not None
        pending = self._pending
        try:
            event = EventRecord(
                pre_frames=pending.pre_frames,
                trigger_frame=pending.trigger_frame,
                post_frames=pending.post_frames,
                hotspot_atoms=pending.hotspot_atoms,
                ood_scores=pending.ood_scores,
                trigger_reason=pending.trigger_reason,
                step=pending.trigger_frame.step,
                time=pending.trigger_frame.time,
                event_id=pending.metadata.get("event_id"),
                backend=pending.metadata.get("backend"),
                model_version=pending.metadata.get("model_version"),
                metadata=pending.metadata,
            )
            self.events.append(event)
        except Exception:
            self._pending = pending
            raise
        self._pending = None
        return event
