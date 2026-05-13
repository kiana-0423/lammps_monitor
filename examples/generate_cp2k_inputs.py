"""Minimal example for CP2K input generation from an extracted region."""

from __future__ import annotations

from pathlib import Path

from hotspot_al.config import load_config
from hotspot_al.extraction.cluster_extractor import extract_cluster_region
from hotspot_al.io.dft_writer import write_dft_inputs
from hotspot_al.io.trajectory_reader import read_trajectory


def main() -> None:
    config = load_config()
    frame = read_trajectory(Path("trajectory.extxyz"))[0]
    region = extract_cluster_region(frame.atoms, [0], config=config)
    written = write_dft_inputs(region, Path("cp2k_inputs"), engine="cp2k", config=config)
    for key, path in written.items():
        print(key, path)


if __name__ == "__main__":
    main()
