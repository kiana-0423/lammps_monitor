"""Online MD monitoring loop that closes LAMMPS, inference, scoring, and events."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from ase.io import write

from hotspot_al.active_learning.scheduler import OnlineEventScheduler
from hotspot_al.buffer.rolling_buffer import RollingBuffer
from hotspot_al.models import EventRecord, FrameData, OODFrameResult
from hotspot_al.monitor.committee_deviation import committee_force_deviation
from hotspot_al.monitor.coordination_monitor import coordination_deltas, smooth_coordination_numbers_fast
from hotspot_al.monitor.force_monitor import delta_force_norms, force_norms
from hotspot_al.monitor.geometry_monitor import displacement_norms, minimum_neighbor_distances_fast
from hotspot_al.monitor.neighbor_utils import MonitorNeighbors
from hotspot_al.monitor.ood_score import OODScorer
from hotspot_al.training.allegro_runner import AllegroRunner
from hotspot_al.exceptions import DataError
from hotspot_al.utils.geometry import row_norms


class FrameSource(Protocol):
    """Small protocol shared by LAMMPSController and fake online sources."""

    def next_frame(self, timeout: float | None = None) -> FrameData | None: ...


EventCallback = Callable[[EventRecord], None]


class OnlineMonitor:
    """Run online OOD monitoring over a live or injected frame source."""

    def __init__(
        self,
        *,
        config: dict[str, Any],
        runner: AllegroRunner,
        frame_source: FrameSource | Iterable[FrameData],
        on_event: EventCallback | None = None,
        scorer: OODScorer | None = None,
        buffer: RollingBuffer | None = None,
        output_dir: str | Path | None = None,
        scheduler: OnlineEventScheduler | None = None,
    ) -> None:
        self.config = config
        self.runner = runner
        self.frame_source = frame_source
        self.scorer = scorer or OODScorer(config)
        buffer_cfg = config.get("buffer", {})
        self.buffer = buffer or RollingBuffer(
            pre_trigger_frames=int(buffer_cfg.get("pre_trigger_frames", 10)),
            post_trigger_frames=int(buffer_cfg.get("post_trigger_frames", 5)),
            maxlen=buffer_cfg.get("maxlen"),
        )
        online_cfg = config.get("online", {})
        self.monitor_freq = max(1, int(online_cfg.get("monitor_freq", 1)))
        self.output_dir = Path(output_dir or online_cfg.get("event_dir") or Path(online_cfg.get("work_dir", ".")) / "events")
        self.on_event = on_event or self.write_event
        self.scheduler = scheduler
        self._previous_positions: np.ndarray | None = None
        self._previous_forces: np.ndarray | None = None
        self._previous_q: np.ndarray | None = None
        self._neighbors: MonitorNeighbors | None = None
        self._last_event_step: int | None = None

    def run(self, *, max_frames: int | None = None, frame_timeout: float | None = None) -> list[OODFrameResult]:
        """Process frames until the source is exhausted or ``max_frames`` is reached."""

        results: list[OODFrameResult] = []
        processed = 0
        try:
            while max_frames is None or processed < max_frames:
                frame = self._next_frame(timeout=frame_timeout)
                if frame is None:
                    break
                result = self.process_frame(frame, frame_index=processed)
                results.append(result)
                event = self.buffer.push(frame)
                finalized_event = event is not None
                if event is not None:
                    self._handle_event(event)
                if result.triggered and not finalized_event and self._last_event_step != frame.step:
                    completed = self.buffer.capture_event(
                        frame,
                        hotspot_atoms=result.hotspot_indices,
                        ood_scores=result.atom_scores,
                        trigger_reason=result.trigger_reason,
                        backend=self.config.get("backend", {}).get("mlip"),
                        model_version=self._model_version(),
                        metadata={"ood": result.metadata},
                    )
                    self._last_event_step = frame.step
                    if completed is not None:
                        self._handle_event(completed)
                processed += 1
        except KeyboardInterrupt:
            pass
        finally:
            event = self.buffer.flush()
            if event is not None:
                self._handle_event(event)
        return results

    def _handle_event(self, event: EventRecord) -> None:
        self.on_event(event)
        if self.scheduler is not None:
            self.scheduler.schedule_event(event)
            self.scheduler.drain()

    def process_frame(self, frame: FrameData, *, frame_index: int = 0) -> OODFrameResult:
        """Compute online metrics and score one frame."""

        if frame.forces is None:
            msg = "Online monitoring requires LAMMPS dump forces fx/fy/fz."
            raise DataError(msg)
        atoms = frame.atoms
        self._ensure_neighbors(atoms)
        if self._neighbors is None:
            msg = "Online monitor neighbor list was not initialized."
            raise DataError(msg)

        q_values = smooth_coordination_numbers_fast(atoms, self._neighbors)
        metrics = {
            "force": force_norms(frame.forces),
            "delta_force": delta_force_norms(frame.forces, self._previous_forces),
            "displacement": displacement_norms(
                atoms.get_positions(),
                self._previous_positions,
                cell=atoms.cell.array,
                pbc=atoms.pbc,
            ),
            "rmin": minimum_neighbor_distances_fast(atoms, self._neighbors),
            "delta_q": coordination_deltas(q_values, self._previous_q),
        }

        full_frame = frame_index % self.monitor_freq == 0
        if full_frame:
            mlip_forces = self.runner.evaluate_forces(atoms, config=self.config, model_path=self._primary_model_path())
            metrics["mlip_force"] = force_norms(mlip_forces)
            metrics["mlip_force_deviation"] = row_norms(np.asarray(frame.forces, dtype=float) - mlip_forces)
            committee_paths = self._committee_model_paths()
            if committee_paths:
                committee_forces = self.runner.evaluate_committee(atoms, config=self.config, model_paths=committee_paths)
                metrics["committee"] = committee_force_deviation(committee_forces)
            result = self.scorer.score_full(metrics, atoms=atoms, forces=frame.forces)
        else:
            result = self.scorer.score_light(metrics)

        self._previous_positions = atoms.get_positions().copy()
        self._previous_forces = np.asarray(frame.forces, dtype=float).copy()
        self._previous_q = q_values.copy()
        return result

    def write_event(self, event: EventRecord) -> None:
        """Default callback: persist event metadata and frames to disk."""

        event_dir = self.output_dir / (event.event_id or f"event-{event.step}")
        event_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "event_id": event.event_id,
            "step": event.step,
            "time": event.time,
            "hotspot_atoms": event.hotspot_atoms,
            "trigger_reason": event.trigger_reason,
            "ood_scores": event.ood_scores.tolist(),
            "backend": event.backend,
            "model_version": event.model_version,
            "metadata": event.metadata,
        }
        (event_dir / "event.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        frames = [*event.pre_frames, event.trigger_frame, *event.post_frames]
        if frames:
            atoms_list = []
            for frame in frames:
                atoms = frame.atoms.copy()
                atoms.info["step"] = frame.step
                if frame.time is not None:
                    atoms.info["time"] = frame.time
                if frame.forces is not None:
                    atoms.arrays["forces"] = np.asarray(frame.forces, dtype=float)
                atoms_list.append(atoms)
            write(event_dir / "frames.extxyz", atoms_list, format="extxyz")

    def _next_frame(self, *, timeout: float | None) -> FrameData | None:
        if hasattr(self.frame_source, "next_frame"):
            return self.frame_source.next_frame(timeout=timeout)  # type: ignore[union-attr]
        if not hasattr(self, "_frame_iterator"):
            self._frame_iterator = iter(self.frame_source)  # type: ignore[arg-type]
        return next(self._frame_iterator, None)

    def _ensure_neighbors(self, atoms: Any) -> None:
        monitor_cfg = self.config.get("monitor", {})
        lj_cutoff = float(monitor_cfg.get("lj_cutoff", 6.0))
        coordination_cutoff = float(monitor_cfg.get("coordination_cutoff", lj_cutoff))
        if self._neighbors is None or self._neighbors.n_atoms != len(atoms):
            self._neighbors = MonitorNeighbors(atoms, lj_cutoff=lj_cutoff, coordination_cutoff=coordination_cutoff)
        else:
            self._neighbors.rebuild(atoms)

    def _primary_model_path(self) -> str | Path | None:
        allegro_cfg = self.config.get("allegro", {})
        deployed = allegro_cfg.get("deployed_model_paths") or []
        model_paths = allegro_cfg.get("model_paths") or []
        if deployed:
            return deployed[0]
        if model_paths:
            return model_paths[0]
        return allegro_cfg.get("checkpoint_path")

    def _committee_model_paths(self) -> list[str]:
        allegro_cfg = self.config.get("allegro", {})
        return list(allegro_cfg.get("deployed_model_paths") or allegro_cfg.get("model_paths") or [])

    def _model_version(self) -> str | None:
        path = self._primary_model_path()
        return None if path is None else Path(path).name
