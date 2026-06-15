"""Minimal example for CP2K input generation from an extracted region."""

from __future__ import annotations

import argparse
from pathlib import Path

from hotspot_al.config import load_config
from hotspot_al.extraction.cluster_extractor import extract_cluster_region
from hotspot_al.io.dft_writer import write_dft_inputs
from hotspot_al.io.trajectory_reader import read_trajectory


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate CP2K inputs from a trajectory frame.")
    parser.add_argument("trajectory", nargs="?", type=Path, default=Path("trajectory.extxyz"), help="Input trajectory")
    parser.add_argument("--output-dir", type=Path, default=Path("cp2k_inputs"), help="Output directory")
    args = parser.parse_args()
    if not args.trajectory.is_file():
        parser.error(f"Trajectory file does not exist: {args.trajectory}")

    config = load_config()
    frame = read_trajectory(args.trajectory)[0]
    region = extract_cluster_region(frame.atoms, [0], config=config)
    written = write_dft_inputs(region, args.output_dir, engine="cp2k", config=config)
    for key, path in written.items():
        print(key, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
