"""Tests for the online monitoring loop without real LAMMPS or Allegro."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

from hotspot_al.models import EventRecord, FrameData
from hotspot_al.monitor.online_monitor import OnlineMonitor
from hotspot_al.training.allegro_runner import AllegroRunner


class CountingEvaluator:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, atoms: Atoms, model_path: str | Path | None, config: dict[str, Any]) -> np.ndarray:
        self.calls += 1
        return np.zeros((len(atoms), 3), dtype=float)


def _frame(step: int, force_scale: float) -> FrameData:
    atoms = Atoms(
        "H4",
        positions=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        cell=[8.0, 8.0, 8.0],
        pbc=False,
    )
    forces = np.zeros((4, 3), dtype=float)
    forces[0, 0] = force_scale
    return FrameData(atoms=atoms, step=step, time=float(step), forces=forces)


def _config() -> dict[str, Any]:
    return {
        "backend": {"mlip": "fake_allegro"},
        "allegro": {"model_paths": ["model-a", "model-b"], "deployed_model_paths": []},
        "monitor": {
            "lj_cutoff": 4.0,
            "delta_q_threshold": 10.0,
            "displacement_z_threshold": 10.0,
        },
        "buffer": {"pre_trigger_frames": 1, "post_trigger_frames": 1, "maxlen": 4},
        "online": {"monitor_freq": 1},
        "ood_score": {
            "weights": {
                "force": 0.0,
                "delta_force": 0.0,
                "rmin": 0.0,
                "delta_q": 0.0,
                "lj_residual": 0.0,
                "committee": 0.0,
                "displacement": 0.0,
                "mlip_force_deviation": 2.0,
            },
            "screen_threshold": 1.0,
            "physics_threshold": 1.0,
            "label_threshold": 1.0,
            "lj_lazy_threshold": 100.0,
            "running_stats": {"warmup_frames": 0, "min_std": 1.0},
        },
    }


def test_online_monitor_triggers_buffer_and_callback() -> None:
    evaluator = CountingEvaluator()
    events: list[EventRecord] = []
    monitor = OnlineMonitor(
        config=_config(),
        runner=AllegroRunner(force_evaluator=evaluator),
        frame_source=[_frame(0, 2.0), _frame(1, 0.1)],
        on_event=events.append,
    )

    results = monitor.run()

    assert len(results) == 2
    assert results[0].triggered
    assert len(events) == 1
    assert events[0].trigger_frame.step == 0
    assert [frame.step for frame in events[0].post_frames] == [1]
    assert evaluator.calls == 6


def test_online_monitor_respects_monitor_frequency() -> None:
    config = _config()
    config["online"]["monitor_freq"] = 2
    evaluator = CountingEvaluator()
    monitor = OnlineMonitor(
        config=config,
        runner=AllegroRunner(force_evaluator=evaluator),
        frame_source=[_frame(0, 2.0), _frame(1, 2.0), _frame(2, 2.0)],
        on_event=lambda _event: None,
    )

    results = monitor.run()

    assert [result.stage for result in results] == ["full", "light", "full"]
    assert evaluator.calls == 6


def test_online_monitor_progress_callback_reports_frames() -> None:
    progress: list[dict[str, Any]] = []
    monitor = OnlineMonitor(
        config=_config(),
        runner=AllegroRunner(force_evaluator=CountingEvaluator()),
        frame_source=[_frame(0, 2.0), _frame(1, 0.1)],
        on_event=lambda _event: None,
        progress_callback=progress.append,
    )

    monitor.run()

    assert [item["processed_frames"] for item in progress] == [1, 2]
    assert progress[-1]["triggered_frames"] >= 1
