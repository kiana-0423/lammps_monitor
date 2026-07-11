"""Scheduler state normalization without requiring Slurm or PBS."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import hotspot_al.backends.schedulers as scheduler_module
from hotspot_al.backends.base import BackendJob, ExecutionRequest, JobState
from hotspot_al.backends.schedulers import PBSSchedulerBackend, SlurmSchedulerBackend


def _job(backend: str) -> BackendJob:
    return BackendJob(backend=backend, state=JobState.SUBMITTED, external_id="42")


@pytest.mark.parametrize("raw, expected", [("RUNNING\n", JobState.RUNNING), ("PENDING\n", JobState.PENDING)])
def test_slurm_uses_live_squeue_state(monkeypatch: pytest.MonkeyPatch, raw: str, expected: JobState) -> None:
    calls: list[list[str]] = []

    def run(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=raw, stderr="")

    monkeypatch.setattr(scheduler_module.subprocess, "run", run)
    assert SlurmSchedulerBackend().poll(_job("slurm")) == expected
    assert len(calls) == 1


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("COMPLETED\n", JobState.COMPLETED),
        ("FAILED\n", JobState.FAILED),
        ("OUT_OF_MEMORY\n", JobState.FAILED),
        ("TIMEOUT\n", JobState.TIMEOUT),
        ("COMPLETED+\n", JobState.COMPLETED),
        ("CANCELLED by 1234\n", JobState.CANCELLED),
    ],
)
def test_slurm_falls_back_to_sacct(monkeypatch: pytest.MonkeyPatch, raw: str, expected: JobState) -> None:
    responses = iter(["", raw])

    def run(command, **_kwargs):
        return subprocess.CompletedProcess(command, 0, stdout=next(responses), stderr="")

    monkeypatch.setattr(scheduler_module.subprocess, "run", run)
    assert SlurmSchedulerBackend().poll(_job("slurm")) == expected


def test_slurm_returns_unknown_when_neither_query_confirms(monkeypatch: pytest.MonkeyPatch) -> None:
    def run(command, **_kwargs):
        return subprocess.CompletedProcess(command, 1 if command[0] == "sacct" else 0, stdout="", stderr="missing")

    monkeypatch.setattr(scheduler_module.subprocess, "run", run)
    assert SlurmSchedulerBackend().poll(_job("slurm")) == JobState.UNKNOWN


def test_slurm_missing_query_commands_return_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(*_args, **_kwargs):
        raise FileNotFoundError("scheduler command")

    monkeypatch.setattr(scheduler_module.subprocess, "run", missing)
    assert SlurmSchedulerBackend().poll(_job("slurm")) == JobState.UNKNOWN


def test_pbs_uses_live_qstat_state(monkeypatch: pytest.MonkeyPatch) -> None:
    output = "Job Id: 42\n    job_state = R\n"
    monkeypatch.setattr(
        scheduler_module.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout=output, stderr=""),
    )
    assert PBSSchedulerBackend().poll(_job("pbs")) == JobState.RUNNING


@pytest.mark.parametrize("exit_status, expected", [(0, JobState.COMPLETED), (17, JobState.FAILED)])
def test_pbs_falls_back_to_history(monkeypatch: pytest.MonkeyPatch, exit_status: int, expected: JobState) -> None:
    calls: list[list[str]] = []

    def run(command, **_kwargs):
        calls.append(command)
        if "-x" not in command:
            return subprocess.CompletedProcess(command, 153, stdout="", stderr="Unknown Job Id")
        output = f"Job Id: 42\n    job_state = F\n    Exit_status = {exit_status}\n"
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    monkeypatch.setattr(scheduler_module.subprocess, "run", run)
    assert PBSSchedulerBackend().poll(_job("pbs")) == expected
    assert calls[1] == ["qstat", "-x", "-f", "42"]


def test_pbs_unsupported_history_returns_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        scheduler_module.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 1, stdout="", stderr="unsupported"),
    )
    assert PBSSchedulerBackend().poll(_job("pbs")) == JobState.UNKNOWN


def test_pbs_missing_query_command_returns_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(*_args, **_kwargs):
        raise FileNotFoundError("qstat")

    monkeypatch.setattr(scheduler_module.subprocess, "run", missing)
    assert PBSSchedulerBackend().poll(_job("pbs")) == JobState.UNKNOWN


def test_pbs_history_query_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def run(command, **_kwargs):
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(command, 153, stdout="", stderr="Unknown Job Id")

    monkeypatch.setattr(scheduler_module.subprocess, "run", run)
    backend = PBSSchedulerBackend({"pbs": {"history_query_enabled": False}})
    assert backend.poll(_job("pbs")) == JobState.UNKNOWN
    assert calls == 1


def test_pbs_script_changes_to_submission_directory(tmp_path: Path) -> None:
    request = ExecutionRequest.from_command(["program"], work_dir=tmp_path)
    text = PBSSchedulerBackend()._render_script(request)
    assert 'cd "$PBS_O_WORKDIR"' in text
