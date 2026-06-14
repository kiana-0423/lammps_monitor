"""Tests for CP2K online task submission."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from ase import Atoms

from hotspot_al.active_learning.scheduler import ScheduledTask
from hotspot_al.config import load_config
from hotspot_al.cp2k.cp2k_task_submitter import CP2KTaskSubmitter
from hotspot_al.models import EventRecord, FrameData
from tests.fake_backends.fake_cp2k import write_fake_cp2k_force_output


class _RunningProcess:
    def __init__(self) -> None:
        self.terminated = False
        self.killed = False

    def poll(self):
        return None

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout=None):
        return 0

    def kill(self) -> None:
        self.killed = True


def _task() -> ScheduledTask:
    atoms = Atoms("CH", positions=[[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]], cell=np.diag([8.0, 8.0, 8.0]), pbc=True)
    frame = FrameData(atoms=atoms, step=12, forces=np.zeros((2, 3)))
    event = EventRecord(
        pre_frames=[],
        trigger_frame=frame,
        post_frames=[],
        hotspot_atoms=[0],
        ood_scores=np.array([7.0, 1.0]),
        trigger_reason=["label_threshold"],
        step=12,
        time=None,
        event_id="evt-cp2k",
    )
    return ScheduledTask(task_id="evt-cp2k", event=event)


def test_cp2k_task_submitter_dry_run_writes_inputs(tmp_path: Path) -> None:
    submitter = CP2KTaskSubmitter(config=load_config(), work_dir=tmp_path, mode="dry_run")
    task = _task()

    submitter(task)

    job = submitter.jobs["evt-cp2k"]
    assert job.status == "prepared"
    assert job.input_file.is_file()
    assert (tmp_path / "evt-cp2k" / "region.extxyz").is_file()
    assert task.metadata["cp2k_job"]["input_file"] == str(job.input_file)


def test_cp2k_task_submitter_poll_writes_dataset(tmp_path: Path) -> None:
    submitter = CP2KTaskSubmitter(config=load_config(), work_dir=tmp_path / "tasks", dataset_dir=tmp_path / "labels", mode="dry_run")
    job = submitter.submit(_task())
    job.mode = "local"
    write_fake_cp2k_force_output(job.output_file, job.metadata["region"].atoms)
    with job.output_file.open("a", encoding="utf-8") as handle:
        handle.write(" SCF run converged\n")

    refreshed = submitter.poll_job("evt-cp2k")

    assert refreshed.status == "completed"
    assert (tmp_path / "labels" / "evt-cp2k.extxyz").is_file()
    assert "dataset_files" in refreshed.metadata


def test_cp2k_task_submitter_marks_timed_out_local_job_failed(tmp_path: Path) -> None:
    config = load_config()
    config["cp2k"]["max_walltime_seconds"] = 0.0
    submitter = CP2KTaskSubmitter(config=config, work_dir=tmp_path, mode="dry_run")
    job = submitter.submit(_task())
    process = _RunningProcess()
    job.mode = "local"
    job.process = process  # type: ignore[assignment]
    job.started_at = 0.0

    refreshed = submitter.poll_job("evt-cp2k")

    assert refreshed.status == "failed"
    assert process.terminated
    assert "walltime" in refreshed.metadata["error"]


def test_cp2k_task_submitter_keeps_running_slurm_job_pending(monkeypatch, tmp_path: Path) -> None:
    submitter = CP2KTaskSubmitter(config=load_config(), work_dir=tmp_path, mode="dry_run")
    job = submitter.submit(_task())
    job.mode = "slurm"
    job.job_id = "12345"

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout="RUNNING\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    refreshed = submitter.poll_job("evt-cp2k")

    assert refreshed.status == "running"
    assert "error" not in refreshed.metadata
