"""LAMMPS input template helpers."""

from __future__ import annotations

from typing import Any


def build_lammps_input(pair_style_block: str, *, config: dict[str, Any]) -> str:
    """Build a minimal LAMMPS input script around a backend pair_style block."""

    timestep_fs = float(config.get("lammps", {}).get("timestep_fs", 0.5))
    return "\n".join(
        [
            "units metal",
            "atom_style atomic",
            "boundary p p p",
            pair_style_block.strip(),
            "",
            f"timestep {timestep_fs}",
            "thermo 100",
        ]
    )
