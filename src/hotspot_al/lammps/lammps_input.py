"""LAMMPS input template helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase import Atoms
from ase.io import write


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


def build_full_lammps_input(
    *,
    pair_style_block: str,
    config: dict[str, Any],
    atoms: Atoms | None = None,
    data_file: str | Path | None = None,
    restart_file: str | Path | None = None,
    output_dir: str | Path | None = None,
    input_name: str = "in.hotspot_al",
) -> str:
    """Build a complete LAMMPS input script for Allegro-driven MD.

    Passing ``atoms`` writes a LAMMPS data file next to the generated input and
    uses ``read_data``. Passing ``restart_file`` uses ``read_restart`` instead.
    """

    if restart_file is not None and (atoms is not None or data_file is not None):
        msg = "Use either restart_file or atoms/data_file, not both."
        raise ValueError(msg)
    if restart_file is None and atoms is None and data_file is None:
        msg = "build_full_lammps_input requires atoms, data_file, or restart_file."
        raise ValueError(msg)

    lammps_cfg = config.get("lammps", {})
    online_cfg = config.get("online", {})
    target_dir = Path(output_dir or online_cfg.get("work_dir", "."))
    target_dir.mkdir(parents=True, exist_ok=True)

    if atoms is not None:
        data_path = Path(data_file or "system.data")
        if not data_path.is_absolute():
            data_path = target_dir / data_path
        write_lammps_data(atoms, data_path, config=config)
        structure_command = f"read_data {data_path.name if data_path.parent == target_dir else data_path}"
    elif restart_file is not None:
        restart_path = Path(restart_file)
        structure_command = f"read_restart {restart_path}"
    else:
        structure_command = f"read_data {data_file}"

    dump_file = str(online_cfg.get("dump_file", "dump.online.lammpstrj"))
    dump_freq = int(online_cfg.get("dump_freq", 10))
    run_steps = int(online_cfg.get("run_steps", lammps_cfg.get("run_steps", 0)))
    thermo_freq = int(lammps_cfg.get("thermo_freq", 100))
    timestep_fs = float(lammps_cfg.get("timestep_fs", 0.5))
    dump_fields = _dump_fields(lammps_cfg)
    template = lammps_cfg.get("full_input_template")
    context = {
        "units": lammps_cfg.get("units", "metal"),
        "atom_style": lammps_cfg.get("atom_style", "atomic"),
        "boundary": lammps_cfg.get("boundary", "p p p"),
        "structure_command": structure_command,
        "pair_style_block": pair_style_block.strip(),
        "timestep_fs": timestep_fs,
        "thermo_freq": thermo_freq,
        "dump_freq": dump_freq,
        "dump_file": dump_file,
        "dump_fields": " ".join(dump_fields),
        "run_command": f"run {run_steps}" if run_steps > 0 else "",
        "input_name": input_name,
    }
    if template:
        return str(template).format(**context)
    lines = [
        f"units {context['units']}",
        f"atom_style {context['atom_style']}",
        f"boundary {context['boundary']}",
        structure_command,
        pair_style_block.strip(),
        "",
        f"timestep {timestep_fs}",
        f"thermo {thermo_freq}",
        f"dump online_dump all custom {dump_freq} {dump_file} {' '.join(dump_fields)}",
        "dump_modify online_dump sort id flush yes",
    ]
    if run_steps > 0:
        lines.append(f"run {run_steps}")
    return "\n".join(lines)


def write_lammps_data(atoms: Atoms, path: str | Path, *, config: dict[str, Any]) -> Path:
    """Write an ASE Atoms object as an atomic LAMMPS data file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    write(target, atoms, format="lammps-data", atom_style=config.get("lammps", {}).get("atom_style", "atomic"))
    return target


def write_full_lammps_input(
    path: str | Path,
    *,
    pair_style_block: str,
    config: dict[str, Any],
    atoms: Atoms | None = None,
    data_file: str | Path | None = None,
    restart_file: str | Path | None = None,
) -> Path:
    """Write a complete LAMMPS input script and return its path."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = build_full_lammps_input(
        pair_style_block=pair_style_block,
        config=config,
        atoms=atoms,
        data_file=data_file,
        restart_file=restart_file,
        output_dir=target.parent,
        input_name=target.name,
    )
    target.write_text(text, encoding="utf-8")
    return target


def _dump_fields(lammps_cfg: dict[str, Any]) -> list[str]:
    fields = list(lammps_cfg.get("dump_fields") or ["id", "type", "x", "y", "z", "fx", "fy", "fz"])
    required = {"id", "type", "x", "y", "z", "fx", "fy", "fz"}
    if not required.issubset(set(fields)):
        fields = ["id", "type", "x", "y", "z", "vx", "vy", "vz", "fx", "fy", "fz"]
    return fields
