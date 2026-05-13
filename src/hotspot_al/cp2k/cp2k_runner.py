"""Runner hooks for preparing and launching CP2K jobs."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def build_cp2k_command(input_file: str | Path, *, config: dict[str, Any]) -> list[str]:
    """Build a CP2K execution command."""

    executable = str(config.get("cp2k", {}).get("executable", "cp2k.popt"))
    return [executable, "-i", str(input_file)]


def run_cp2k(input_file: str | Path, *, config: dict[str, Any], dry_run: bool = True) -> list[str] | subprocess.CompletedProcess[str]:
    """Run CP2K or return the command in dry-run mode."""

    command = build_cp2k_command(input_file, config=config)
    if dry_run:
        return command
    return subprocess.run(command, check=True, text=True, capture_output=True)
