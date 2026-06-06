"""Run an online monitoring loop with optional CP2K dry-run scheduling."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

from hotspot_al.active_learning.scheduler import OnlineEventScheduler
from hotspot_al.config import load_config
from hotspot_al.cp2k.cp2k_task_submitter import CP2KTaskSubmitter
from hotspot_al.lammps.allegro_lammps import AllegroBackend
from hotspot_al.lammps.lammps_controller import LAMMPSController
from hotspot_al.models import FrameData
from hotspot_al.monitor.online_monitor import OnlineMonitor
from hotspot_al.training.allegro_runner import AllegroRunner


def fake_force_evaluator(atoms: Atoms, model_path: str | Path | None, config: dict[str, Any]) -> np.ndarray:
    """Deterministic stand-in for real Allegro inference."""

    return np.zeros((len(atoms), 3), dtype=float)


def fake_frames() -> list[FrameData]:
    """Small injected frame source for trying the Python loop without LAMMPS."""

    atoms = Atoms("H4", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], cell=[8, 8, 8], pbc=False)
    frames = []
    for step, force in enumerate([2.0, 0.1]):
        forces = np.zeros((len(atoms), 3), dtype=float)
        forces[0, 0] = force
        frames.append(FrameData(atoms=atoms.copy(), step=step, time=float(step), forces=forces))
    return frames


def main() -> None:
    config = load_config()
    config["online"]["enabled"] = True
    config["online"]["event_dir"] = "./lammps_run/events"
    config["cp2k"]["submit_mode"] = "dry_run"

    cp2k_submitter = CP2KTaskSubmitter(config=config, mode="dry_run")
    scheduler = OnlineEventScheduler(submitter=cp2k_submitter)
    runner = AllegroRunner(force_evaluator=fake_force_evaluator)

    source: Any = fake_frames()

    if os.environ.get("HOTSPOT_AL_REAL_LAMMPS") == "1":
        backend = AllegroBackend(config=config)
        pair_block = "\n".join(backend.write_lammps_input().splitlines()[3:5])
        initial_atoms = source[0].atoms
        source = LAMMPSController.from_atoms(
            initial_atoms,
            pair_style_block=pair_block,
            config=config,
            work_dir=config["online"]["work_dir"],
        )

    monitor = OnlineMonitor(config=config, runner=runner, frame_source=source, scheduler=scheduler)
    results = monitor.run(max_frames=100)

    print(f"processed_frames={len(results)}")
    print(f"scheduled_cp2k_tasks={len(scheduler.submitted)}")


if __name__ == "__main__":
    main()
