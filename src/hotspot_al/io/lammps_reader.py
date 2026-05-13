"""Unified LAMMPS trajectory readers."""

from __future__ import annotations

from pathlib import Path

from hotspot_al.lammps.dump_parser import iter_lammps_dump, read_lammps_dump
from hotspot_al.models import FrameData


def read_dump(
    path: str | Path,
    *,
    type_map: dict[int, str] | None = None,
    timestep_fs: float | None = None,
) -> list[FrameData]:
    """Read a LAMMPS dump file into ``FrameData`` objects."""

    return read_lammps_dump(path, type_map=type_map, timestep_fs=timestep_fs)


def iter_dump(
    path: str | Path,
    *,
    type_map: dict[int, str] | None = None,
    timestep_fs: float | None = None,
):
    """Iterate over ``FrameData`` parsed from a LAMMPS dump file."""

    return iter_lammps_dump(path, type_map=type_map, timestep_fs=timestep_fs)
