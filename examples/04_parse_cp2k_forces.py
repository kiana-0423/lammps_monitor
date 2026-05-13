"""Parse CP2K forces from an output file."""

from __future__ import annotations

from pathlib import Path

from hotspot_al.cp2k.cp2k_force_parser import parse_cp2k_forces


def main() -> None:
    forces = parse_cp2k_forces(Path("cp2k.out"))
    print(forces.shape)


if __name__ == "__main__":
    main()
