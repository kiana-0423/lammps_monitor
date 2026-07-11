"""Built-in Local, Slurm, and PBS scheduler adapters."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

from hotspot_al.backends.base import (
    BackendJob,
    BackendRole,
    ExecutionRequest,
    JobState,
    RuntimeStatus,
    SchedulerBackend,
)


class LocalSchedulerBackend(SchedulerBackend):
    backend_name = "local"
    role = BackendRole.SCHEDULER

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = dict(config or {})

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "LocalSchedulerBackend":
        return cls(config)

    def check_runtime(self) -> RuntimeStatus:
        return RuntimeStatus(self.backend_name, self.role, True, "local subprocess runtime available")

    def submit(self, request: ExecutionRequest) -> BackendJob:
        request.work_dir.mkdir(parents=True, exist_ok=True)
        stdout_handle = None if request.stdout_path is None else _open_output(request.work_dir, request.stdout_path)
        stderr_handle = None if request.stderr_path is None else _open_output(request.work_dir, request.stderr_path)
        environment = {**os.environ, **request.environment}
        try:
            process = subprocess.Popen(
                list(request.command),
                cwd=request.work_dir,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                env=environment,
            )
        finally:
            if stdout_handle is not None:
                stdout_handle.close()
            if stderr_handle is not None:
                stderr_handle.close()
        return BackendJob(
            backend=self.backend_name,
            state=JobState.SUBMITTED,
            external_id=str(process.pid),
            native_handle=process,
            metadata=dict(request.metadata),
        )

    def poll(self, job: BackendJob) -> JobState:
        process = job.native_handle
        if not isinstance(process, subprocess.Popen):
            return job.state
        returncode = process.poll()
        if returncode is None:
            job.state = JobState.RUNNING
        else:
            job.state = JobState.COMPLETED if returncode == 0 else JobState.FAILED
            job.metadata["returncode"] = returncode
        return job.state

    def cancel(self, job: BackendJob) -> None:
        process = job.native_handle
        if isinstance(process, subprocess.Popen) and process.poll() is None:
            process.terminate()
        job.state = JobState.CANCELLED


class _BatchSchedulerBackend(SchedulerBackend):
    submit_command: str
    query_command: str
    cancel_command: str
    directive_prefix: str
    script_name: str

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = dict(config or {})

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "_BatchSchedulerBackend":
        return cls(config)

    def check_runtime(self) -> RuntimeStatus:
        resolved = shutil.which(self.submit_command)
        return RuntimeStatus(
            self.backend_name,
            self.role,
            resolved is not None,
            resolved or f"scheduler command not found: {self.submit_command}",
        )

    def submit(self, request: ExecutionRequest) -> BackendJob:
        request.work_dir.mkdir(parents=True, exist_ok=True)
        script = request.work_dir / self.script_name
        script.write_text(self._render_script(request), encoding="utf-8")
        result = subprocess.run(
            [self.submit_command, str(script)],
            cwd=request.work_dir,
            check=True,
            text=True,
            capture_output=True,
        )
        external_id = self._parse_job_id(result.stdout)
        return BackendJob(
            backend=self.backend_name,
            state=JobState.SUBMITTED,
            external_id=external_id,
            metadata={**dict(request.metadata), "script": str(script)},
        )

    def cancel(self, job: BackendJob) -> None:
        if job.external_id:
            subprocess.run([self.cancel_command, job.external_id], check=False, text=True, capture_output=True)
        job.state = JobState.CANCELLED

    def _render_script(self, request: ExecutionRequest) -> str:
        lines = ["#!/bin/bash"]
        lines.extend(self._resource_directives(request.resources))
        lines.append("set -euo pipefail")
        lines.extend(self._working_directory_lines())
        lines.extend(f"export {key}={shlex.quote(value)}" for key, value in request.environment.items())
        command = shlex.join(request.command)
        if request.stdout_path is not None:
            command += f" > {shlex.quote(str(request.stdout_path))}"
        if request.stderr_path is not None:
            command += f" 2> {shlex.quote(str(request.stderr_path))}"
        lines.extend([command, ""])
        return "\n".join(lines)

    def _working_directory_lines(self) -> list[str]:
        return []

    def _resource_directives(self, resources: Mapping[str, Any]) -> list[str]:
        raw = resources.get("directives", "")
        if isinstance(raw, str):
            return [line for line in raw.splitlines() if line.strip()]
        return [str(line) for line in raw]

    def _parse_job_id(self, output: str) -> str:
        parts = output.strip().split()
        return parts[-1] if parts else uuid4().hex


class SlurmSchedulerBackend(_BatchSchedulerBackend):
    backend_name = "slurm"
    role = BackendRole.SCHEDULER
    submit_command = "sbatch"
    query_command = "squeue"
    cancel_command = "scancel"
    directive_prefix = "#SBATCH"
    script_name = "submit.sbatch"

    def poll(self, job: BackendJob) -> JobState:
        if not job.external_id:
            return JobState.UNKNOWN
        try:
            result = subprocess.run(
                [self.query_command, "-h", "-j", job.external_id, "-o", "%T"],
                check=False,
                text=True,
                capture_output=True,
            )
        except FileNotFoundError:
            result = None
        state = _first_status(result.stdout) if result is not None and result.returncode == 0 else None
        if state is None:
            try:
                result = subprocess.run(
                    ["sacct", "-n", "-X", "-j", job.external_id, "--format=State"],
                    check=False,
                    text=True,
                    capture_output=True,
                )
            except FileNotFoundError:
                job.state = JobState.UNKNOWN
                return job.state
            state = _first_status(result.stdout) if result.returncode == 0 else None
        job.state = _slurm_state(state)
        return job.state


class PBSSchedulerBackend(_BatchSchedulerBackend):
    backend_name = "pbs"
    role = BackendRole.SCHEDULER
    submit_command = "qsub"
    query_command = "qstat"
    cancel_command = "qdel"
    directive_prefix = "#PBS"
    script_name = "submit.pbs"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(config)
        pbs_config = self.config.get("pbs", {})
        self.history_query_enabled = bool(pbs_config.get("history_query_enabled", True))

    def _working_directory_lines(self) -> list[str]:
        return ['cd "$PBS_O_WORKDIR"']

    def poll(self, job: BackendJob) -> JobState:
        if not job.external_id:
            return JobState.UNKNOWN
        try:
            result = subprocess.run(
                [self.query_command, "-f", job.external_id], check=False, text=True, capture_output=True
            )
        except FileNotFoundError:
            job.state = JobState.UNKNOWN
            return job.state
        output = result.stdout if result.returncode == 0 else ""
        if not output and self.history_query_enabled:
            try:
                result = subprocess.run(
                    [self.query_command, "-x", "-f", job.external_id], check=False, text=True, capture_output=True
                )
            except FileNotFoundError:
                result = None
            if result is not None and result.returncode == 0:
                output = result.stdout
        job.state = _pbs_state(output)
        return job.state


def _first_status(output: str) -> str | None:
    for line in output.splitlines():
        if line.strip():
            return line.strip().split()[0].rstrip("+").upper()
    return None


def _slurm_state(state: str | None) -> JobState:
    normalized = "" if state is None else state.split()[0].rstrip("+").upper()
    return {
        "PENDING": JobState.PENDING,
        "CONFIGURING": JobState.PENDING,
        "RUNNING": JobState.RUNNING,
        "COMPLETING": JobState.RUNNING,
        "COMPLETED": JobState.COMPLETED,
        "CANCELLED": JobState.CANCELLED,
        "FAILED": JobState.FAILED,
        "OUT_OF_MEMORY": JobState.FAILED,
        "NODE_FAIL": JobState.FAILED,
        "PREEMPTED": JobState.FAILED,
        "TIMEOUT": JobState.TIMEOUT,
    }.get(normalized, JobState.UNKNOWN)


def _pbs_state(output: str) -> JobState:
    fields: dict[str, str] = {}
    for line in output.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key.strip().lower()] = value.strip()
    state = fields.get("job_state", "").upper()
    if state in {"Q", "H", "W", "S"}:
        return JobState.PENDING
    if state in {"R", "E", "B"}:
        return JobState.RUNNING
    if state == "F":
        try:
            return JobState.COMPLETED if int(fields.get("exit_status", "1")) == 0 else JobState.FAILED
        except ValueError:
            return JobState.UNKNOWN
    if state == "C":
        return JobState.COMPLETED
    return JobState.UNKNOWN


def _open_output(work_dir: Path, path: Path):
    resolved = path if path.is_absolute() else work_dir / path
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved.open("w", encoding="utf-8")


__all__ = ["LocalSchedulerBackend", "PBSSchedulerBackend", "SlurmSchedulerBackend"]
