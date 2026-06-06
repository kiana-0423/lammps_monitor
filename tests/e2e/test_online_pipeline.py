"""Online E2E protocol test using fake inference and dry-run CP2K."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

from hotspot_al.active_learning.scheduler import OnlineEventScheduler
from hotspot_al.config import load_config
from hotspot_al.cp2k.cp2k_task_submitter import CP2KTaskSubmitter
from hotspot_al.models import FrameData
from hotspot_al.monitor.online_monitor import OnlineMonitor
from hotspot_al.training.allegro_runner import AllegroRunner
from tests.fake_backends.fake_allegro import fake_force_evaluator


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


def _online_config(tmp_path: Path) -> dict[str, Any]:
    config = load_config()
    config["allegro"]["model_paths"] = ["model-a"]
    config["online"] = {
        **config["online"],
        "work_dir": str(tmp_path / "lammps"),
        "event_dir": str(tmp_path / "events"),
        "monitor_freq": 2,
    }
    config["buffer"] = {"pre_trigger_frames": 1, "post_trigger_frames": 1, "maxlen": 4}
    config["monitor"] = {
        **config["monitor"],
        "lj_cutoff": 4.0,
        "delta_q_threshold": 10.0,
        "displacement_z_threshold": 10.0,
    }
    config["ood_score"] = {
        **config["ood_score"],
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
        "running_stats": {"enabled": True, "warmup_frames": 0, "min_std": 1.0},
    }
    config["extraction"] = {**config["extraction"], "min_atoms": 4, "max_atoms": 8, "extract_radius": 4.0}
    return config


def test_online_pipeline_writes_event_and_schedules_cp2k(tmp_path: Path) -> None:
    config = _online_config(tmp_path)
    submitter = CP2KTaskSubmitter(config=config, work_dir=tmp_path / "cp2k", mode="dry_run")
    scheduler = OnlineEventScheduler(submitter=submitter)
    monitor = OnlineMonitor(
        config=config,
        runner=AllegroRunner(force_evaluator=fake_force_evaluator),
        frame_source=[_frame(0, 2.0), _frame(1, 0.1), _frame(2, 0.1)],
        scheduler=scheduler,
    )

    results = monitor.run()

    assert [result.stage for result in results] == ["full", "light", "full"]
    assert scheduler.submitted
    task_id = scheduler.submitted[0].task_id
    assert (tmp_path / "events" / task_id / "event.json").is_file()
    assert (tmp_path / "events" / task_id / "frames.extxyz").is_file()
    assert submitter.jobs[task_id].status == "prepared"
    assert submitter.jobs[task_id].input_file.is_file()
