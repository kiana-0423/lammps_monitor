"""Generate CP2K inputs for an extracted hotspot region."""

from __future__ import annotations

from pathlib import Path

from hotspot_al.config import load_config
from hotspot_al.cp2k.cp2k_input import write_cp2k_inputs
from hotspot_al.extraction.cluster_extractor import extract_cluster_region
from hotspot_al.io.trajectory_reader import read_trajectory


def main() -> None:
    config = load_config()
    frame = read_trajectory(Path("trajectory.extxyz"))[0]
    region = extract_cluster_region(frame.atoms, [0], config=config)
    written = write_cp2k_inputs(region, Path("cp2k_jobs"), config=config)
    print(written)


if __name__ == "__main__":
    main()
