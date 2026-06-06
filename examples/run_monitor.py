"""Minimal example for atom-wise monitoring and OOD scoring."""

from __future__ import annotations

from pathlib import Path

from hotspot_al.config import load_config
from hotspot_al.monitor.coordination_monitor import coordination_deltas, smooth_coordination_numbers_fast
from hotspot_al.monitor.force_monitor import delta_force_norms, force_norms
from hotspot_al.monitor.geometry_monitor import minimum_neighbor_distances_fast
from hotspot_al.monitor.neighbor_utils import MonitorNeighbors
from hotspot_al.monitor.ood_score import OODScorer


def main() -> None:
    from hotspot_al.io.trajectory_reader import iter_trajectory

    config = load_config()
    scorer = OODScorer(config)
    previous_forces = None
    previous_q = None
    neighbors = None

    trajectory_path = Path("trajectory.extxyz")
    for frame in iter_trajectory(trajectory_path):
        if frame.forces is None:
            raise ValueError("This example expects per-atom forces in the trajectory.")
        monitor_cfg = config["monitor"]
        if neighbors is None:
            neighbors = MonitorNeighbors(
                frame.atoms,
                lj_cutoff=monitor_cfg.get("lj_cutoff", 6.0),
                coordination_cutoff=monitor_cfg.get("coordination_cutoff", monitor_cfg.get("lj_cutoff", 6.0)),
            )
        else:
            neighbors.rebuild(frame.atoms)
        q_values = smooth_coordination_numbers_fast(frame.atoms, neighbors)
        metrics = {
            "force": force_norms(frame.forces),
            "delta_force": delta_force_norms(frame.forces, previous_forces),
            "rmin": minimum_neighbor_distances_fast(frame.atoms, neighbors),
            "delta_q": coordination_deltas(q_values, previous_q),
        }
        result = scorer.score_frame(metrics)
        print(frame.step, result.max_score, result.hotspot_indices, result.trigger_reason)
        previous_forces = frame.forces
        previous_q = q_values


if __name__ == "__main__":
    main()
