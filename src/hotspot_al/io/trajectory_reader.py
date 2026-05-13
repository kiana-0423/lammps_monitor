"""Trajectory reading utilities with a unified ``FrameData`` interface."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
from ase import Atoms
from ase.io import iread

from hotspot_al.lammps.dump_parser import iter_lammps_dump
from hotspot_al.models import FrameData


def _infer_ase_format(path: Path, fmt: str | None = None) -> str | None:
    if fmt is not None:
        return fmt
    suffix = path.suffix.lower()
    if suffix in {".lammpstrj", ".dump"}:
        return "lammps-dump-text"
    return None


def _extract_velocities(atoms: Atoms) -> np.ndarray | None:
    try:
        velocities = atoms.get_velocities()
    except Exception:
        return None
    if velocities is None:
        return None
    return np.asarray(velocities, dtype=float)


def _extract_forces(atoms: Atoms) -> np.ndarray | None:
    if "forces" in atoms.arrays:
        return np.asarray(atoms.arrays["forces"], dtype=float)
    try:
        return np.asarray(atoms.get_forces(apply_constraint=False), dtype=float)
    except Exception:
        return None


def _load_external_forces(path: str | Path) -> np.ndarray:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".npy":
        forces = np.load(source)
    elif suffix == ".npz":
        archive = np.load(source)
        key = "forces" if "forces" in archive.files else archive.files[0]
        forces = archive[key]
    elif suffix in {".txt", ".dat", ".csv"}:
        forces = np.loadtxt(source, delimiter="," if suffix == ".csv" else None)
    else:
        msg = f"Unsupported force file format: {source}"
        raise ValueError(msg)
    forces = np.asarray(forces, dtype=float)
    if forces.ndim not in {2, 3}:
        msg = f"External force array must be 2D or 3D, received {forces.shape}"
        raise ValueError(msg)
    return forces


def frame_from_atoms(atoms: Atoms, step: int, external_forces: np.ndarray | None = None) -> FrameData:
    """Convert an ASE ``Atoms`` object to ``FrameData``."""

    metadata = {
        "symbols": atoms.get_chemical_symbols(),
        "cell": atoms.cell.array.copy(),
        "pbc": atoms.pbc.copy(),
    }
    if "time" in atoms.info:
        metadata["time"] = atoms.info["time"]
    if "step" in atoms.info:
        metadata["source_step"] = atoms.info["step"]
    return FrameData(
        atoms=atoms.copy(),
        step=step,
        time=float(atoms.info["time"]) if "time" in atoms.info else None,
        forces=external_forces if external_forces is not None else _extract_forces(atoms),
        velocities=_extract_velocities(atoms),
        metadata=metadata,
    )


def iter_trajectory(
    path: str | Path,
    *,
    fmt: str | None = None,
    index: str | slice | None = ":",
    force_path: str | Path | None = None,
    type_map: dict[int, str] | None = None,
    timestep_fs: float | None = None,
) -> Iterator[FrameData]:
    """Iterate over a trajectory and yield ``FrameData`` objects."""

    source = Path(path)
    ase_format = _infer_ase_format(source, fmt)
    if ase_format == "lammps-dump-text":
        yield from iter_lammps_dump(source, type_map=type_map, timestep_fs=timestep_fs)
        return
    external_forces = _load_external_forces(force_path) if force_path is not None else None
    frames = iread(source, format=ase_format, index=index)
    for step, atoms in enumerate(frames):
        forces = None
        if external_forces is not None:
            if external_forces.ndim == 2:
                forces = external_forces
            else:
                forces = external_forces[step]
        yield frame_from_atoms(atoms, step=step, external_forces=forces)


def read_trajectory(
    path: str | Path,
    *,
    fmt: str | None = None,
    index: str | slice | None = ":",
    force_path: str | Path | None = None,
    type_map: dict[int, str] | None = None,
    timestep_fs: float | None = None,
) -> list[FrameData]:
    """Read a whole trajectory into memory."""

    return list(
        iter_trajectory(
            path,
            fmt=fmt,
            index=index,
            force_path=force_path,
            type_map=type_map,
            timestep_fs=timestep_fs,
        )
    )
