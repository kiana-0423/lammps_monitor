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


def build_online_lammps_input(
    *,
    data_file: str,
    pair_style_block: str,
    config: dict[str, Any],
) -> str:
    """Build a runnable LAMMPS script for online dump monitoring."""

    lammps_cfg = config.get("lammps", {})
    online_cfg = config.get("online", {})
    timestep_fs = float(lammps_cfg.get("timestep_fs", 0.5))
    dump_file = str(online_cfg.get("dump_file", "dump.online.lammpstrj"))
    dump_freq = int(online_cfg.get("dump_freq", 10))
    run_steps = int(online_cfg.get("run_steps", 0))
    dump_fields = lammps_cfg.get("dump_fields") or ["id", "type", "x", "y", "z", "fx", "fy", "fz"]
    if not {"id", "type", "x", "y", "z", "fx", "fy", "fz"}.issubset(set(dump_fields)):
        dump_fields = ["id", "type", "x", "y", "z", "vx", "vy", "vz", "fx", "fy", "fz"]
    lines = [
        "units metal",
        "atom_style atomic",
        "boundary p p p",
        f"read_data {data_file}",
        pair_style_block.strip(),
        "",
        f"timestep {timestep_fs}",
        "thermo 100",
        f"dump online_dump all custom {dump_freq} {dump_file} {' '.join(dump_fields)}",
        "dump_modify online_dump sort id flush yes",
    ]
    if run_steps > 0:
        lines.append(f"run {run_steps}")
    return "\n".join(lines)
