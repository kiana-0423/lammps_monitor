"""Minimal example for hotspot extraction from a triggered frame."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from hotspot_al.config import load_config
from hotspot_al.extraction.cluster_extractor import extract_cluster_region
from hotspot_al.hotspot.hotspot_detector import detect_hotspots
from hotspot_al.io.trajectory_reader import read_trajectory


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect a synthetic hotspot and extract its local cluster.")
    parser.add_argument("trajectory", nargs="?", type=Path, default=Path("trajectory.extxyz"), help="Input trajectory")
    args = parser.parse_args()
    if not args.trajectory.is_file():
        parser.error(f"Trajectory file does not exist: {args.trajectory}")

    config = load_config()
    frames = read_trajectory(args.trajectory)
    frame = frames[0]
    fake_scores = np.zeros(len(frame.atoms))
    fake_scores[0] = 7.0
    hotspots = detect_hotspots(
        frame.atoms,
        fake_scores,
        threshold=config["hotspot"].get("threshold", 6.0),
        merge_radius=config["hotspot"]["merge_radius"],
        step=frame.step,
        trigger_reasons=["force_burst"],
    )
    if not hotspots:
        raise RuntimeError("No hotspots detected from the synthetic scores.")
    region = extract_cluster_region(frame.atoms, hotspots[0].core_atom_indices, config=config)
    print("extracted atoms:", len(region.atoms))
    print("core:", region.core_indices)
    print("boundary:", region.boundary_indices)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
