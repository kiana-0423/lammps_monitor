"""LAMMPS process control and streaming dump reader."""

from __future__ import annotations

import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from ase import Atoms

from hotspot_al.lammps.dump_parser import iter_lammps_dump
from hotspot_al.lammps.lammps_input import write_full_lammps_input
from hotspot_al.lammps.lammps_runner import build_lammps_command
from hotspot_al.models import FrameData
from hotspot_al.exceptions import LAMMPSRuntimeError
from hotspot_al.utils.logging import configure_logging


class LAMMPSController:
    """Launch LAMMPS and yield complete custom-dump frames as they appear."""

    def __init__(
        self,
        input_file: str | Path,
        *,
        dump_file: str | Path,
        config: dict[str, Any],
        work_dir: str | Path | None = None,
        poll_interval: float = 0.1,
        stderr_file: str | Path | None = None,
    ) -> None:
        self.input_file = Path(input_file)
        self.config = config
        self.work_dir = Path(work_dir or self.input_file.parent)
        dump_path = Path(dump_file)
        self.dump_file = dump_path if dump_path.is_absolute() else self.work_dir / dump_path
        self.poll_interval = float(poll_interval)
        self.stderr_file = Path(stderr_file) if stderr_file is not None else self.work_dir / "lammps.stderr.log"
        self.process: subprocess.Popen[str] | None = None
        self.logger = configure_logging(config, name=__name__)
        self._offset = 0
        self._queued_frames: list[FrameData] = []

    @classmethod
    def from_atoms(
        cls,
        atoms: Atoms,
        *,
        pair_style_block: str,
        config: dict[str, Any],
        work_dir: str | Path | None = None,
        input_name: str = "in.hotspot_al",
        data_name: str = "system.data",
        poll_interval: float = 0.1,
    ) -> "LAMMPSController":
        """Write a complete LAMMPS run directory and return a controller."""

        resolved_work_dir = Path(work_dir or config.get("online", {}).get("work_dir", "."))
        input_file = resolved_work_dir / input_name
        write_full_lammps_input(
            input_file,
            pair_style_block=pair_style_block,
            config=config,
            atoms=atoms,
            data_file=data_name,
        )
        dump_file = config.get("online", {}).get("dump_file", "dump.online.lammpstrj")
        return cls(
            input_file,
            dump_file=dump_file,
            config=config,
            work_dir=resolved_work_dir,
            poll_interval=poll_interval,
        )

    def __enter__(self) -> "LAMMPSController":
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    def start(self) -> None:
        """Start LAMMPS in the configured working directory."""

        if self.process is not None and self.process.poll() is None:
            return
        self.work_dir.mkdir(parents=True, exist_ok=True)
        command = build_lammps_command(self.input_file, config=self.config)
        self.logger.info("starting LAMMPS command=%s work_dir=%s", command, self.work_dir)
        stderr_handle = self.stderr_file.open("a", encoding="utf-8")
        self.process = subprocess.Popen(
            command,
            cwd=self.work_dir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_handle,
            text=True,
        )

    def stop(self, *, timeout: float = 10.0) -> None:
        """Terminate LAMMPS gracefully, escalating only if it does not exit."""

        if self.process is None or self.process.poll() is not None:
            return
        self.logger.info("stopping LAMMPS pid=%s", self.process.pid)
        self.process.terminate()
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.logger.warning("LAMMPS did not stop within %.1fs; killing pid=%s", timeout, self.process.pid)
            self.process.kill()
            self.process.wait(timeout=timeout)
        self.logger.info("LAMMPS stopped with returncode=%s", self.process.returncode)

    def pause(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.send_signal(signal.SIGSTOP)

    def resume(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.send_signal(signal.SIGCONT)

    def next_frame(self, timeout: float | None = None) -> FrameData | None:
        """Wait for the next complete dump frame, or ``None`` when finished."""

        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if self._queued_frames:
                return self._queued_frames.pop(0)
            frames = self._read_new_frames()
            if frames:
                self._queued_frames.extend(frames[1:])
                return frames[0]
            if self.process is not None and self.process.poll() is not None:
                self.assert_healthy()
                return None
            if deadline is not None and time.monotonic() >= deadline:
                return None
            time.sleep(self.poll_interval)

    def is_running(self) -> bool:
        """Return whether the managed LAMMPS process is still alive."""

        return self.process is not None and self.process.poll() is None

    def assert_healthy(self) -> None:
        """Raise if LAMMPS exited with a non-zero return code."""

        if self.process is None:
            return
        returncode = self.process.poll()
        if returncode not in (None, 0):
            msg = f"LAMMPS exited with returncode={returncode}; see {self.stderr_file}"
            raise LAMMPSRuntimeError(msg)

    def _read_new_frames(self) -> list[FrameData]:
        if not self.dump_file.exists():
            return []
        text = self.dump_file.read_text(encoding="utf-8")
        if self._offset >= len(text):
            return []
        chunk = text[self._offset :]
        markers = [index for index in range(len(chunk)) if chunk.startswith("ITEM: TIMESTEP", index)]
        if len(markers) < 2:
            if self.process is not None and self.process.poll() is not None and chunk.strip():
                complete = chunk
                self._offset = len(text)
            else:
                return []
        else:
            complete = chunk[: markers[-1]]
            self._offset += markers[-1]
        return list(
            iter_lammps_dump(
                _DumpText(complete),
                type_map=self.config.get("lammps", {}).get("type_map"),
                timestep_fs=self.config.get("lammps", {}).get("timestep_fs"),
            )
        )


class _DumpText:
    """Tiny Path-like shim so existing dump parser can parse in-memory text."""

    def __init__(self, text: str) -> None:
        self._text = text

    def read_text(self, encoding: str = "utf-8") -> str:
        return self._text
