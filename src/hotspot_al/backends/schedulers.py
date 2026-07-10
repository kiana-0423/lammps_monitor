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
        lines.extend(f"export {key}={shlex.quote(value)}" for key, value in request.environment.items())
        command = shlex.join(request.command)
        if request.stdout_path is not None:
            command += f" > {shlex.quote(str(request.stdout_path))}"
        if request.stderr_path is not None:
            command += f" 2> {shlex.quote(str(request.stderr_path))}"
        lines.extend([command, ""])
        return "\n".join(lines)

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
        result = subprocess.run(
            [self.query_command, "-h", "-j", job.external_id, "-o", "%T"],
            check=False,
            text=True,
            capture_output=True,
        )
        raw = result.stdout.strip().splitlines()
        if not raw:
            job.state = JobState.UNKNOWN
            return job.state
        state = raw[0].strip().upper()
        job.state = {
            "PENDING": JobState.PENDING,
            "RUNNING": JobState.RUNNING,
            "COMPLETED": JobState.COMPLETED,
            "CANCELLED": JobState.CANCELLED,
            "FAILED": JobState.FAILED,
            "TIMEOUT": JobState.FAILED,
        }.get(state, JobState.UNKNOWN)
        return job.state


class PBSSchedulerBackend(_BatchSchedulerBackend):
    backend_name = "pbs"
    role = BackendRole.SCHEDULER
    submit_command = "qsub"
    query_command = "qstat"
    cancel_command = "qdel"
    directive_prefix = "#PBS"
    script_name = "submit.pbs"

    def poll(self, job: BackendJob) -> JobState:
        if not job.external_id:
            return JobState.UNKNOWN
        result = subprocess.run([self.query_command, job.external_id], check=False, text=True, capture_output=True)
        if result.returncode != 0:
            job.state = JobState.UNKNOWN
            return job.state
        output = result.stdout.upper()
        if " R " in output:
            job.state = JobState.RUNNING
        elif " Q " in output or " H " in output:
            job.state = JobState.PENDING
        elif " C " in output:
            job.state = JobState.COMPLETED
        else:
            job.state = JobState.UNKNOWN
        return job.state


def _open_output(work_dir: Path, path: Path):
    resolved = path if path.is_absolute() else work_dir / path
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved.open("w", encoding="utf-8")


__all__ = ["LocalSchedulerBackend", "PBSSchedulerBackend", "SlurmSchedulerBackend"]
