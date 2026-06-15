"""Write one hotspot region to an Allegro extxyz dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from hotspot_al.config import load_config
from hotspot_al.extraction.cluster_extractor import extract_cluster_region
from hotspot_al.io.trajectory_reader import read_trajectory
from hotspot_al.training.allegro_adapter import write_allegro_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Write one extracted hotspot region to an Allegro dataset.")
    parser.add_argument("trajectory", nargs="?", type=Path, default=Path("trajectory.extxyz"), help="Input trajectory")
    parser.add_argument("--output-dir", type=Path, default=Path("allegro_data"), help="Output dataset directory")
    args = parser.parse_args()
    if not args.trajectory.is_file():
        parser.error(f"Trajectory file does not exist: {args.trajectory}")

    config = load_config()
    frame = read_trajectory(args.trajectory)[0]
    region = extract_cluster_region(frame.atoms, [0], config=config)
    forces = np.zeros((len(region.atoms), 3), dtype=float)
    written = write_allegro_dataset(region, forces=forces, output_dir=args.output_dir, config=config)
    print(written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
