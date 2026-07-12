"""Submit online hotspot events as CP2K labeling tasks."""

from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ase.io import write

from hotspot_al.active_learning.scheduler import ScheduledTask
from hotspot_al.backends.base import BackendJob, DFTBackend, JobState, SchedulerBackend
from hotspot_al.backends.factory import create_typed_backend
from hotspot_al.datasets import write_dataset_entry
from hotspot_al.extraction.block import BlockCooldownTracker, extract_block_regions
from hotspot_al.extraction.cluster_extractor import extract_cluster_region
from hotspot_al.extraction.h_capping import add_h_caps
from hotspot_al.models import ExtractedRegion
from hotspot_al.training.mask_generator import generate_atom_mask
from hotspot_al.utils.logging import configure_logging


@dataclass(slots=True)
class CP2KSubmittedJob:
    """Bookkeeping for one CP2K labeling submission."""

    task_id: str
    work_dir: Path
    input_file: Path
    output_file: Path
    mode: str
    process: subprocess.Popen[str] | None = None
    job_id: str | None = None
    status: str = "submitted"
    attempts: int = 1
    started_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    scheduler_job: BackendJob | None = None


class CP2KTaskSubmitter:
    """TaskSubmitter-compatible CP2K labeling backend.

    The submitter is intentionally conservative: every task gets a stable
    directory, generated CP2K inputs, and a metadata file. Dry-run mode stops
    there. Local and Slurm modes submit the single-point input and can be
    followed up through ``poll_job``.
    """

    def __init__(
        self,
        *,
        config: dict[str, Any],
        work_dir: str | Path | None = None,
        dataset_dir: str | Path | None = None,
        mode: str | None = None,
        max_retries: int | None = None,
        dft_backend: DFTBackend | None = None,
        scheduler_backend: SchedulerBackend | None = None,
    ) -> None:
        cp2k_cfg = config.get("cp2k", {})
        self.config = config
        self.work_dir = Path(work_dir or cp2k_cfg.get("task_dir", "./cp2k_tasks"))
        self.dataset_dir = Path(dataset_dir or cp2k_cfg.get("labeled_dataset_dir", "./labeled_data"))
        self.mode = str(mode or cp2k_cfg.get("submit_mode", "dry_run"))
        self.max_retries = int(max_retries if max_retries is not None else cp2k_cfg.get("max_retries", 0))
        self.max_walltime_seconds = _resolve_walltime_seconds(config)
        self.logger = configure_logging(config, name=__name__)
        if self.mode not in {"dry_run", "local", "slurm", "pbs"}:
            msg = "DFT submit mode must be one of: dry_run, local, slurm, pbs."
            raise ValueError(msg)
        self.dft_backend = dft_backend or create_typed_backend(config, "dft", DFTBackend)
        self.scheduler_backend = scheduler_backend
        if self.scheduler_backend is None and self.mode != "dry_run":
            effective_config = {**config, "backend": {**config.get("backend", {}), "scheduler": {"engine": self.mode}}}
            self.scheduler_backend = create_typed_backend(effective_config, "scheduler", SchedulerBackend)
        self.jobs: dict[str, CP2KSubmittedJob] = {}
        block_cfg = config.get("extraction", {}).get("block", {})
        self.block_cooldown = BlockCooldownTracker(int(block_cfg.get("cooldown_steps", 0)))

    def __call__(self, task: ScheduledTask) -> None:
        """Prepare and submit one scheduled event task."""

        try:
            job = self.submit(task)
        except Exception:
            self.logger.exception("CP2K task submission failed for %s", task.task_id)
            task.metadata["cp2k_job"] = {"task_id": task.task_id, "status": "failed"}
            raise
        task.metadata["cp2k_job"] = self._job_payload(job)
        if "additional_jobs" in job.metadata:
            task.metadata["cp2k_jobs"] = [task.metadata["cp2k_job"], *job.metadata["additional_jobs"]]

    def submit(self, task: ScheduledTask) -> CP2KSubmittedJob:
        """Create CP2K inputs for ``task`` and submit according to mode."""

        self.logger.info("preparing CP2K task %s", task.task_id)
        regions = self._prepare_regions(task)
        jobs = [
            self._submit_region(
                task,
                region,
                task_id=task.task_id if index == 0 else f"{task.task_id}__region-{index + 1}",
                region_index=index,
                region_count=len(regions),
            )
            for index, region in enumerate(regions)
        ]
        primary = jobs[0]
        if len(jobs) > 1:
            primary.metadata["additional_jobs"] = [self._job_payload(job) for job in jobs[1:]]
        return primary

    def _submit_region(
        self,
        task: ScheduledTask,
        region: ExtractedRegion,
        *,
        task_id: str,
        region_index: int,
        region_count: int,
    ) -> CP2KSubmittedJob:
        task_dir = self.work_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        write(task_dir / "region.extxyz", region.atoms, format="extxyz")
        written = self.dft_backend.prepare_inputs(region, task_dir, task_id=task_id)
        input_file = written.get("single_point_input") or next(iter(written.values()))
        output_file = task_dir / f"{Path(input_file).stem}.out"
        metadata = {
            "task_id": task_id,
            "source_task_id": task.task_id,
            "region_index": region_index,
            "region_count": region_count,
            "event_step": task.event.step,
            "hotspot_atoms": task.event.hotspot_atoms,
            "region_atoms": len(region.atoms),
            "input_files": {key: str(path) for key, path in written.items()},
        }
        (task_dir / "task.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        job = CP2KSubmittedJob(
            task_id=task_id,
            work_dir=task_dir,
            input_file=Path(input_file),
            output_file=output_file,
            mode=self.mode,
            status="prepared" if self.mode == "dry_run" else "submitted",
            metadata={"region": region, **metadata},
        )
        if self.mode in {"local", "slurm", "pbs"}:
            self._submit_with_scheduler(job)
        self.jobs[task_id] = job
        self.logger.info("CP2K task %s status=%s mode=%s", task_id, job.status, job.mode)
        return job

    def poll_job(self, task_id: str) -> CP2KSubmittedJob:
        """Refresh job status and write training data when CP2K has completed."""

        job = self.jobs[task_id]
        if job.mode == "dry_run":
            return job
        if job.scheduler_job is not None and self.scheduler_backend is not None:
            scheduler_state = self.scheduler_backend.poll(job.scheduler_job)
            if scheduler_state in {JobState.SUBMITTED, JobState.PENDING, JobState.RUNNING}:
                if self._timed_out(job):
                    self.scheduler_backend.cancel(job.scheduler_job)
                    job.status = "failed"
                    job.metadata["error"] = f"DFT task exceeded max walltime of {self.max_walltime_seconds:.1f}s."
                    return self._retry_or_return(job)
                job.status = "running" if scheduler_state == JobState.RUNNING else "pending"
                return job
            if scheduler_state in {JobState.FAILED, JobState.CANCELLED}:
                job.status = "failed"
                job.metadata["error"] = f"Scheduler reported {scheduler_state.value}."
                return self._retry_or_return(job)
        if job.process is not None and job.process.poll() is None:
            if self._timed_out(job):
                self.logger.warning("CP2K task %s exceeded walltime; terminating", task_id)
                job.process.terminate()
                try:
                    job.process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    job.process.kill()
                    job.process.wait(timeout=5.0)
                job.status = "failed"
                job.metadata["error"] = f"CP2K task exceeded max walltime of {self.max_walltime_seconds:.1f}s."
                return self._retry_or_return(job)
            job.status = "running"
            return job
        if job.mode == "slurm" and job.job_id is not None and not job.output_file.exists():
            slurm_status = self._query_slurm_status(job.job_id)
            if slurm_status in {"PENDING", "RUNNING"}:
                job.status = slurm_status.lower()
                return job
            if slurm_status is not None:
                job.metadata["slurm_status"] = slurm_status
        if not job.output_file.exists():
            job.status = "failed"
            job.metadata["error"] = f"CP2K output was not produced: {job.output_file}"
            self.logger.warning("CP2K task %s failed: %s", task_id, job.metadata["error"])
            return self._retry_or_return(job)
        output_text = job.output_file.read_text(encoding="utf-8", errors="ignore")
        if not self.dft_backend.output_is_complete(output_text):
            job.status = "failed"
            job.metadata["error"] = "CP2K output did not contain an SCF convergence marker."
            self.logger.warning("CP2K task %s failed: %s", task_id, job.metadata["error"])
            return self._retry_or_return(job)
        region = job.metadata["region"]
        forces = self.dft_backend.parse_forces(job.output_file)
        mask = generate_atom_mask(region, self.config)
        written = write_dataset_entry(
            region,
            forces=forces,
            mask=mask,
            output_dir=self.dataset_dir,
            prefix=job.task_id,
            extra_metadata={"cp2k_output": str(job.output_file)},
        )
        job.metadata["dataset_files"] = {key: str(path) for key, path in written.items()}
        job.status = "completed"
        self.logger.info("CP2K task %s completed; dataset files written to %s", task_id, self.dataset_dir)
        return job

    def _prepare_region(self, task: ScheduledTask) -> ExtractedRegion:
        return self._prepare_regions(task)[0]

    def _prepare_regions(self, task: ScheduledTask) -> list[ExtractedRegion]:
        frame = task.event.trigger_frame
        if str(self.config.get("extraction", {}).get("mode", "cluster")) == "block":
            regions = extract_block_regions(
                frame.atoms,
                task.event.hotspot_atoms,
                config=self.config,
                step=frame.step,
                cooldown_tracker=self.block_cooldown,
            )
            if not regions:
                msg = f"no block region selected for task {task.task_id}; all candidate blocks may be in cooldown."
                raise ValueError(msg)
        else:
            region = extract_cluster_region(frame.atoms, task.event.hotspot_atoms, config=self.config)
            region = add_h_caps(frame.atoms, region, config=self.config)
            regions = [region]
        for index, region in enumerate(regions):
            region.metadata = {
                **region.metadata,
                "original_frame_id": frame.step,
                "hotspot_id": task.task_id if index == 0 else f"{task.task_id}__region-{index + 1}",
                "event_id": task.event.event_id,
                "source_hotspot_id": task.task_id,
                "region_index": index,
                "region_count": len(regions),
            }
        return regions

    def _submit_local(self, job: CP2KSubmittedJob) -> subprocess.Popen[str]:
        request = self.dft_backend.execution_request(job.input_file, output_file=job.output_file)
        command = list(request.command)
        self.logger.info("submitting local CP2K task %s command=%s", job.task_id, command)
        stdout_handle = job.output_file.open("w", encoding="utf-8")
        stderr_handle = (job.work_dir / f"{job.input_file.stem}.err").open("w", encoding="utf-8")
        job.started_at = time.monotonic()
        return subprocess.Popen(
            command,
            cwd=job.work_dir,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )

    def _submit_slurm(self, job: CP2KSubmittedJob) -> str | None:
        request = self.dft_backend.execution_request(job.input_file, output_file=job.output_file)
        command = " ".join([*request.command, ">", job.output_file.name])
        script = "\n".join(
            [
                "#!/bin/bash",
                f"#SBATCH --job-name={job.task_id}",
                f"#SBATCH --output={job.task_id}.slurm.out",
                self.config.get("cp2k", {}).get("slurm_directives", "").strip(),
                "set -euo pipefail",
                command,
                "",
            ]
        )
        script_path = job.work_dir / "submit.sbatch"
        script_path.write_text(script, encoding="utf-8")
        self.logger.info("submitting Slurm CP2K task %s with %s", job.task_id, script_path)
        result = subprocess.run(["sbatch", str(script_path)], cwd=job.work_dir, check=True, text=True, capture_output=True)
        parts = result.stdout.strip().split()
        return parts[-1] if parts else None

    def _query_slurm_status(self, job_id: str) -> str | None:
        try:
            result = subprocess.run(
                ["squeue", "-h", "-j", job_id, "-o", "%T"],
                check=False,
                text=True,
                capture_output=True,
            )
        except FileNotFoundError:
            return None
        status = result.stdout.strip().splitlines()
        return status[0].strip() if status else None

    def _retry_or_return(self, job: CP2KSubmittedJob) -> CP2KSubmittedJob:
        if job.mode != "local" or job.attempts > self.max_retries:
            return job
        job.attempts += 1
        retry_input = _input_with_adjusted_scf(job.input_file, attempt=job.attempts)
        job.input_file = retry_input
        job.output_file = job.work_dir / f"{retry_input.stem}.out"
        self.logger.info("retrying CP2K task %s attempt=%d", job.task_id, job.attempts)
        if self.scheduler_backend is not None:
            self._submit_with_scheduler(job)
        else:
            job.process = self._submit_local(job)
        job.status = "submitted"
        return job

    def _submit_with_scheduler(self, job: CP2KSubmittedJob) -> None:
        if self.scheduler_backend is None:
            raise RuntimeError(f"No scheduler backend configured for mode {job.mode!r}.")
        request = self.dft_backend.execution_request(job.input_file, output_file=job.output_file)
        resources = {"directives": self.config.get("cp2k", {}).get("slurm_directives", "")}
        request = type(request)(
            command=request.command,
            work_dir=request.work_dir,
            stdout_path=request.stdout_path,
            stderr_path=request.stderr_path,
            environment=request.environment,
            resources=resources,
            metadata={**request.metadata, "task_id": job.task_id},
        )
        scheduler_job = self.scheduler_backend.submit(request)
        job.scheduler_job = scheduler_job
        job.process = scheduler_job.native_handle if isinstance(scheduler_job.native_handle, subprocess.Popen) else None
        job.job_id = scheduler_job.external_id
        job.started_at = time.monotonic()

    def _timed_out(self, job: CP2KSubmittedJob) -> bool:
        return (
            self.max_walltime_seconds is not None
            and job.started_at is not None
            and time.monotonic() - job.started_at > self.max_walltime_seconds
        )

    def _job_payload(self, job: CP2KSubmittedJob) -> dict[str, Any]:
        return {
            "task_id": job.task_id,
            "mode": job.mode,
            "status": job.status,
            "work_dir": str(job.work_dir),
            "input_file": str(job.input_file),
            "output_file": str(job.output_file),
            "job_id": job.job_id,
        }


def _looks_converged(output_text: str) -> bool:
    markers = ("SCF run converged", "SCF converged", "ENERGY|")
    return any(marker in output_text for marker in markers)


def _resolve_walltime_seconds(config: dict[str, Any]) -> float | None:
    cp2k_cfg = config.get("cp2k", {})
    raw = cp2k_cfg.get("max_walltime_seconds", cp2k_cfg.get("max_walltime", config.get("online", {}).get("max_walltime")))
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    parts = str(raw).split(":")
    try:
        if len(parts) == 3:
            hours, minutes, seconds = (float(part) for part in parts)
            return hours * 3600 + minutes * 60 + seconds
        if len(parts) == 2:
            minutes, seconds = (float(part) for part in parts)
            return minutes * 60 + seconds
        return float(raw)
    except ValueError:
        return None


def _input_with_adjusted_scf(input_file: Path, *, attempt: int) -> Path:
    text = input_file.read_text(encoding="utf-8")
    multiplier = max(2, attempt)

    def replace(match: re.Match[str]) -> str:
        value = int(match.group("value"))
        return f"{match.group('prefix')}{value * multiplier}"

    adjusted = re.sub(r"(?m)^(?P<prefix>\s*MAX_SCF\s+)(?P<value>\d+)\s*$", replace, text)
    retry_path = input_file.with_name(f"{input_file.stem}_retry{attempt}{input_file.suffix}")
    retry_path.write_text(adjusted, encoding="utf-8")
    return retry_path
