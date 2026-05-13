"""Extract hotspot-localized regions from a scored frame."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hotspot_al.active_learning.workflow import extract_regions_for_result
from hotspot_al.config import load_config
from hotspot_al.io.trajectory_reader import read_trajectory
from hotspot_al.models import OODFrameResult


def main() -> None:
    config = load_config()
    frame = read_trajectory(Path("trajectory.extxyz"))[0]
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


if __name__ == "__main__":
    main()
