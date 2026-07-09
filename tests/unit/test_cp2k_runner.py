"""Tests for CP2K command helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path

from hotspot_al.cp2k.cp2k_runner import build_cp2k_command, run_cp2k


def test_build_cp2k_command_uses_configured_executable() -> None:
    command = build_cp2k_command(Path("input.inp"), config={"cp2k": {"executable": "cp2k.psmp"}})

    assert command == ["cp2k.psmp", "-i", "input.inp"]


def test_run_cp2k_dry_run_returns_command() -> None:
    command = run_cp2k("toy.inp", config={"cp2k": {"executable": "cp2k.popt"}}, dry_run=True)

    assert command == ["cp2k.popt", "-i", "toy.inp"]


def test_run_cp2k_executes_subprocess_when_not_dry_run(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_run(command, **kwargs):
        calls.append({"command": command, **kwargs})
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_cp2k("toy.inp", config={"cp2k": {"executable": "cp2k.ssmp"}}, dry_run=False)

    assert isinstance(result, subprocess.CompletedProcess)
    assert result.stdout == "ok"
    assert calls == [
        {
            "command": ["cp2k.ssmp", "-i", "toy.inp"],
            "check": True,
            "text": True,
            "capture_output": True,
        }
    ]
