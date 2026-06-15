"""Parsing helpers for LAMMPS custom dump trajectories."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Protocol

import numpy as np
from ase import Atoms

from hotspot_al.exceptions import DataError
from hotspot_al.models import FrameData


class SupportsReadText(Protocol):
    """Minimal interface shared by Path and in-memory dump shims."""

    def read_text(self, encoding: str = "utf-8") -> str: ...


def _parse_box_bounds(header: str, lines: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(lines) < 3:
        msg = "Malformed LAMMPS dump: truncated BOX BOUNDS section."
        raise DataError(msg)
    tokens = header.split()[3:]
    boundary_tokens = [token for token in tokens if token in {"pp", "ff", "fs", "sf", "ss", "fm", "mf"}]
    pbc = np.array([token.startswith("p") for token in boundary_tokens[:3]], dtype=bool)
    if len(pbc) < 3:
        pbc = np.array([True, True, True], dtype=bool)

    first = [float(value) for value in lines[0].split()]
    second = [float(value) for value in lines[1].split()]
    third = [float(value) for value in lines[2].split()]
    if len(first) == 2:
        xlo, xhi = first
        ylo, yhi = second
        zlo, zhi = third
        cell = np.array(
            [
                [xhi - xlo, 0.0, 0.0],
                [0.0, yhi - ylo, 0.0],
                [0.0, 0.0, zhi - zlo],
            ],
            dtype=float,
        )
        origin = np.array([xlo, ylo, zlo], dtype=float)
        return cell, origin, pbc

    xlo_bound, xhi_bound, xy = first
    ylo_bound, yhi_bound, xz = second
    zlo_bound, zhi_bound, yz = third
    xlo = xlo_bound - min(0.0, xy, xz, xy + xz)
    xhi = xhi_bound - max(0.0, xy, xz, xy + xz)
    ylo = ylo_bound - min(0.0, yz)
    yhi = yhi_bound - max(0.0, yz)
    zlo = zlo_bound
    zhi = zhi_bound
    cell = np.array(
        [
            [xhi - xlo, 0.0, 0.0],
            [xy, yhi - ylo, 0.0],
            [xz, yz, zhi - zlo],
        ],
        dtype=float,
    )
    origin = np.array([xlo, ylo, zlo], dtype=float)
    return cell, origin, pbc


def _get_positions(fields: list[str], table: dict[str, np.ndarray]) -> np.ndarray:
    if {"x", "y", "z"}.issubset(fields):
        return np.column_stack([table["x"], table["y"], table["z"]])
    if {"xu", "yu", "zu"}.issubset(fields):
        return np.column_stack([table["xu"], table["yu"], table["zu"]])
    if {"xs", "ys", "zs"}.issubset(fields):
        return np.column_stack([table["xs"], table["ys"], table["zs"]])
    msg = "LAMMPS dump must contain x/y/z, xu/yu/zu, or xs/ys/zs coordinates."
    raise DataError(msg)


def _symbol_list(table: dict[str, np.ndarray], type_map: dict[int, str] | None) -> list[str]:
    if "element" in table:
        return [str(value) for value in table["element"]]
    if "type" not in table:
        msg = "LAMMPS dump must contain either 'element' or 'type'."
        raise DataError(msg)
    if type_map is None:
        msg = "A type_map is required when the dump does not contain 'element'."
        raise DataError(msg)
    try:
        return [type_map[int(value)] for value in table["type"]]
    except KeyError as exc:
        msg = f"LAMMPS type {exc.args[0]} is missing from type_map."
        raise DataError(msg) from exc


def parse_lammps_dump_frame(
    step: int,
    cell: np.ndarray,
    origin: np.ndarray,
    pbc: np.ndarray,
    fields: list[str],
    atom_lines: list[str],
    *,
    type_map: dict[int, str] | None = None,
    timestep_fs: float | None = None,
) -> FrameData:
    """Parse one LAMMPS dump frame into ``FrameData``."""

    raw_rows = [line.split() for line in atom_lines]
    for row in raw_rows:
        if len(row) < len(fields):
            msg = "Malformed LAMMPS dump: atom row has fewer columns than ATOMS header."
            raise DataError(msg)
    table: dict[str, np.ndarray] = {}
    for field_index, field_name in enumerate(fields):
        column = [row[field_index] for row in raw_rows]
        try:
            if field_name == "element":
                table[field_name] = np.asarray(column, dtype=object)
            else:
                table[field_name] = np.asarray(column, dtype=float)
        except ValueError as exc:
            msg = f"Malformed LAMMPS dump: non-numeric values in column {field_name!r}."
            raise DataError(msg) from exc

    order = np.argsort(table["id"]) if "id" in table else np.arange(len(atom_lines))
    for name, values in list(table.items()):
        table[name] = values[order]

    positions = _get_positions(fields, table)
    if {"xs", "ys", "zs"}.issubset(fields):
        positions = positions @ cell
    positions = positions - origin

    symbols = _symbol_list(table, type_map)
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=pbc)

    velocities = None
    if {"vx", "vy", "vz"}.issubset(fields):
        velocities = np.column_stack([table["vx"], table["vy"], table["vz"]])
        atoms.set_velocities(velocities)

    forces = None
    if {"fx", "fy", "fz"}.issubset(fields):
        forces = np.column_stack([table["fx"], table["fy"], table["fz"]])
        atoms.arrays["forces"] = forces

    energy = float(np.sum(table["c_pe"])) if "c_pe" in table else None
    metadata: dict[str, Any] = {
        "ids": table["id"].astype(int).tolist() if "id" in table else list(range(1, len(atoms) + 1)),
        "fields": fields,
        "origin": origin.tolist(),
    }
    if "type" in table:
        metadata["types"] = table["type"].astype(int).tolist()
    if "q" in table:
        metadata["charges"] = table["q"].astype(float).tolist()
    if "mol" in table:
        metadata["molecule_ids"] = table["mol"].astype(int).tolist()
    return FrameData(
        atoms=atoms,
        step=step,
        time=None if timestep_fs is None else step * float(timestep_fs),
        forces=forces,
        velocities=velocities,
        energy=energy,
        metadata=metadata,
    )


def iter_lammps_dump(
    path: str | Path | SupportsReadText,
    *,
    type_map: dict[int, str] | None = None,
    timestep_fs: float | None = None,
) -> Iterator[FrameData]:
    """Iterate over a LAMMPS custom dump file."""

    if hasattr(path, "read_text"):
        text = path.read_text(encoding="utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    lines = text.splitlines()
    cursor = 0
    while cursor < len(lines):
        if not lines[cursor].startswith("ITEM: TIMESTEP"):
            cursor += 1
            continue
        if cursor + 8 >= len(lines):
            msg = "Malformed LAMMPS dump: truncated frame header."
            raise DataError(msg)
        try:
            step = int(lines[cursor + 1].strip())
        except ValueError as exc:
            msg = "Malformed LAMMPS dump: invalid timestep."
            raise DataError(msg) from exc
        if not lines[cursor + 2].startswith("ITEM: NUMBER OF ATOMS"):
            msg = "Malformed LAMMPS dump: missing NUMBER OF ATOMS section."
            raise DataError(msg)
        try:
            n_atoms = int(lines[cursor + 3].strip())
        except ValueError as exc:
            msg = "Malformed LAMMPS dump: invalid atom count."
            raise DataError(msg) from exc
        box_header = lines[cursor + 4]
        cell, origin, pbc = _parse_box_bounds(box_header, lines[cursor + 5 : cursor + 8])
        atoms_header = lines[cursor + 8]
        if not atoms_header.startswith("ITEM: ATOMS"):
            msg = "Malformed LAMMPS dump: missing ATOMS section."
            raise DataError(msg)
        fields = atoms_header.split()[2:]
        atom_lines = lines[cursor + 9 : cursor + 9 + n_atoms]
        if len(atom_lines) != n_atoms:
            msg = "Malformed LAMMPS dump: truncated atom section."
            raise DataError(msg)
        yield parse_lammps_dump_frame(
            step,
            cell,
            origin,
            pbc,
            fields,
            atom_lines,
            type_map=type_map,
            timestep_fs=timestep_fs,
        )
        cursor = cursor + 9 + n_atoms


def read_lammps_dump(
    path: str | Path,
    *,
    type_map: dict[int, str] | None = None,
    timestep_fs: float | None = None,
) -> list[FrameData]:
    """Read all frames from a LAMMPS custom dump file."""

    return list(iter_lammps_dump(path, type_map=type_map, timestep_fs=timestep_fs))
