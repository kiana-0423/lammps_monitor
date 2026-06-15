"""Generate CP2K inputs for an extracted hotspot region."""

from __future__ import annotations

import argparse
from pathlib import Path

from hotspot_al.config import load_config
from hotspot_al.cp2k.cp2k_input import write_cp2k_inputs
from hotspot_al.extraction.cluster_extractor import extract_cluster_region
from hotspot_al.io.trajectory_reader import read_trajectory


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate CP2K inputs for the first atom in a trajectory frame.")
    parser.add_argument("trajectory", nargs="?", type=Path, default=Path("trajectory.extxyz"), help="Input trajectory")
    parser.add_argument("--output-dir", type=Path, default=Path("cp2k_jobs"), help="Directory for generated CP2K inputs")
    args = parser.parse_args()
    if not args.trajectory.is_file():
        parser.error(f"Trajectory file does not exist: {args.trajectory}")

    config = load_config()
    frame = read_trajectory(args.trajectory)[0]
    region = extract_cluster_region(frame.atoms, [0], config=config)
    written = write_cp2k_inputs(region, args.output_dir, config=config)
    print(written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
