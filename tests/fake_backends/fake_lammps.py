"""Fake LAMMPS fixture helpers."""

from __future__ import annotations

from pathlib import Path

TOY_DUMP = """ITEM: TIMESTEP
100
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type x y z fx fy fz
1 1 1.0 1.0 1.0 1.0 0.0 0.0
2 3 2.0 1.0 1.0 0.0 0.0 0.0
"""


def write_fake_lammps_dump(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(TOY_DUMP, encoding="utf-8")
    return target

