"""Tests for LAMMPS streaming dump control helpers."""

from __future__ import annotations

import pytest

import hotspot_al.lammps.lammps_controller as controller_module
from hotspot_al.exceptions import LAMMPSRuntimeError
from hotspot_al.lammps.lammps_controller import LAMMPSController

FRAME0 = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS pp pp pp
0 5
0 5
0 5
ITEM: ATOMS id type x y z fx fy fz
1 1 0 0 0 1 0 0
"""

FRAME1 = """ITEM: TIMESTEP
1
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS pp pp pp
0 5
0 5
0 5
ITEM: ATOMS id type x y z fx fy fz
1 1 0.1 0 0 0.5 0 0
"""


def test_lammps_controller_waits_for_complete_streaming_frame(tmp_path) -> None:
    dump_file = tmp_path / "dump.online.lammpstrj"
    controller = LAMMPSController(
        tmp_path / "in.lammps",
        dump_file=dump_file,
        work_dir=tmp_path,
        config={"lammps": {"type_map": {1: "H"}, "timestep_fs": 0.5}},
        poll_interval=0.0,
    )

    dump_file.write_text(FRAME0, encoding="utf-8")
    assert controller.next_frame(timeout=0.0) is None

    dump_file.write_text(FRAME0 + FRAME1, encoding="utf-8")
    frame = controller.next_frame(timeout=0.0)

    assert frame is not None
    assert frame.step == 0
    assert len(frame.atoms) == 1
    assert frame.forces[0, 0] == 1.0


class _ExitedProcess:
    def poll(self) -> int:
        return 2


def test_lammps_controller_reports_nonzero_process_exit(tmp_path) -> None:
    controller = LAMMPSController(
        tmp_path / "in.lammps",
        dump_file=tmp_path / "dump.online.lammpstrj",
        work_dir=tmp_path,
        config={"lammps": {"type_map": {1: "H"}, "timestep_fs": 0.5}},
        poll_interval=0.0,
    )
    controller.process = _ExitedProcess()  # type: ignore[assignment]

    with pytest.raises(LAMMPSRuntimeError, match="returncode=2"):
        controller.next_frame(timeout=0.0)


class _RunningProcess:
    pid = 123
    returncode = 0

    def __init__(self, *_args, **_kwargs) -> None:
        self._running = True

    def poll(self) -> int | None:
        return None if self._running else self.returncode

    def terminate(self) -> None:
        self._running = False

    def wait(self, timeout: float | None = None) -> int:
        self._running = False
        return self.returncode

    def kill(self) -> None:
        self._running = False


def test_lammps_controller_closes_log_handles_on_stop(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(controller_module.subprocess, "Popen", _RunningProcess)
    controller = LAMMPSController(
        tmp_path / "in.lammps",
        dump_file=tmp_path / "dump.online.lammpstrj",
        work_dir=tmp_path,
        config={"lammps": {"executable": "fake-lmp", "type_map": {1: "H"}, "timestep_fs": 0.5}},
        poll_interval=0.0,
    )

    controller.start()
    assert controller._stdout_handle is not None
    assert not controller._stdout_handle.closed

    controller.stop()

    assert controller._stdout_handle is None
    assert controller._stderr_handle is None
