"""Runner hooks for launching LAMMPS simulations."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def build_lammps_command(input_file: str | Path, *, config: dict[str, Any]) -> list[str]:
    """Build a LAMMPS execution command."""

    executable = str(config.get("lammps", {}).get("executable", "lmp"))
    return [executable, "-in", str(input_file)]


def run_lammps(input_file: str | Path, *, config: dict[str, Any], dry_run: bool = True) -> list[str] | subprocess.CompletedProcess[str]:
    """Run LAMMPS or return the command in dry-run mode."""

    command = build_lammps_command(input_file, config=config)
    if dry_run:
        return command
    return subprocess.run(command, check=True, text=True, capture_output=True)
