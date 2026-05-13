"""Write one hotspot region to an Allegro extxyz dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hotspot_al.config import load_config
from hotspot_al.extraction.cluster_extractor import extract_cluster_region
from hotspot_al.io.trajectory_reader import read_trajectory
from hotspot_al.training.allegro_adapter import write_allegro_dataset


def main() -> None:
    config = load_config()
    frame = read_trajectory(Path("trajectory.extxyz"))[0]
    region = extract_cluster_region(frame.atoms, [0], config=config)
    forces = np.zeros((len(region.atoms), 3), dtype=float)
    written = write_allegro_dataset(region, forces=forces, output_dir=Path("allegro_data"), config=config)
    print(written)


if __name__ == "__main__":
    main()
