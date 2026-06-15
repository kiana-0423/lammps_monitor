"""Parse CP2K forces from an output file."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from hotspot_al.cp2k.cp2k_force_parser import parse_cp2k_forces


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse atomic forces from a CP2K output file.")
    parser.add_argument("output", nargs="?", type=Path, default=Path("cp2k.out"), help="CP2K .out file")
    args = parser.parse_args()

    if not args.output.is_file():
        parser.error(f"CP2K output file does not exist: {args.output}")

    forces = parse_cp2k_forces(args.output)
    print(forces.shape)
    print("min:", np.min(forces, axis=0))
    print("max:", np.max(forces, axis=0))
    print("mean:", np.mean(forces, axis=0))
    print("norm_min:", float(np.min(np.linalg.norm(forces, axis=1))))
    print("norm_max:", float(np.max(np.linalg.norm(forces, axis=1))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
