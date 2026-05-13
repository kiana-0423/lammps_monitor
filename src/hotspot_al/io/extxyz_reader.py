"""Convenience readers and writers for extxyz datasets."""

from __future__ import annotations

from pathlib import Path

from ase.io import iread, write

from hotspot_al.io.trajectory_reader import frame_from_atoms
from hotspot_al.models import FrameData


def read_extxyz(path: str | Path) -> list[FrameData]:
    """Read an extxyz file into ``FrameData`` objects."""

    return [frame_from_atoms(atoms, step=index) for index, atoms in enumerate(iread(path, format="extxyz"))]


def write_extxyz(path: str | Path, frames: list[FrameData]) -> None:
    """Write ``FrameData`` objects to one extxyz file."""

    atoms_list = []
    for frame in frames:
        atoms = frame.atoms.copy()
        if frame.forces is not None:
            atoms.arrays["forces"] = frame.forces
        if frame.velocities is not None:
            atoms.set_velocities(frame.velocities)
        atoms.info["step"] = frame.step
        if frame.time is not None:
            atoms.info["time"] = frame.time
        atoms_list.append(atoms)
    write(path, atoms_list, format="extxyz")
