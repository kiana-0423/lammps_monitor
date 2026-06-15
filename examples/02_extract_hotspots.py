"""Extract hotspot-localized regions from a scored frame."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from hotspot_al.active_learning.workflow import extract_regions_for_result
from hotspot_al.config import load_config
from hotspot_al.io.trajectory_reader import read_trajectory
from hotspot_al.models import OODFrameResult


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract hotspot-localized regions from a trajectory frame.")
    parser.add_argument("trajectory", nargs="?", type=Path, default=Path("trajectory.extxyz"), help="Input trajectory")
    args = parser.parse_args()
    if not args.trajectory.is_file():
        parser.error(f"Trajectory file does not exist: {args.trajectory}")

    config = load_config()
    frame = read_trajectory(args.trajectory)[0]
    scores = np.zeros(len(frame.atoms))
    scores[0] = 7.5
    result = OODFrameResult(
        atom_scores=scores,
        metric_scores={"force": scores},
        max_score=7.5,
        hotspot_indices=[0],
        trigger_reason=["force_large"],
        triggered=True,
        metadata={"backend": config["backend"]["mlip"], "event_id": "evt-demo"},
    )
    regions = extract_regions_for_result(frame, result, config=config)
    print("n_regions:", len(regions))
    print("n_atoms first region:", len(regions[0].atoms))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
