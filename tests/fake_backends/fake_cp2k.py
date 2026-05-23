"""Fake CP2K output writer for offline parser tests."""

from __future__ import annotations

from pathlib import Path

from ase import Atoms


def write_fake_cp2k_force_output(path: str | Path, atoms: Atoms) -> Path:
    """Write a minimal CP2K force block parseable by the project parser."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        " ATOMIC FORCES in [a.u.]",
        "",
        " #  Atom   Kind   Element          X              Y              Z",
    ]
    for index, symbol in enumerate(atoms.get_chemical_symbols(), start=1):
        fx = 0.001 * index
        fy = -0.0005 * index
        fz = 0.00025 * index
        lines.append(f"{index:7d}      1      {symbol:<2} {fx:16.10f} {fy:16.10f} {fz:16.10f}")
    lines.append(" SUM OF ATOMIC FORCES          0.0            0.0            0.0")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target

