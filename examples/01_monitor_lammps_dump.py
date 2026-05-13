"""Monitor a LAMMPS dump with atom-wise PHAL metrics."""

from __future__ import annotations

from pathlib import Path

from hotspot_al.config import load_config
from hotspot_al.io.lammps_reader import iter_dump
from hotspot_al.monitor.coordination_monitor import coordination_deltas, smooth_coordination_numbers
from hotspot_al.monitor.force_monitor import delta_force_norms, force_norms
from hotspot_al.monitor.geometry_monitor import displacement_norms, minimum_neighbor_distances
from hotspot_al.monitor.ood_score import OODScorer


def main() -> None:
    config = load_config()
    scorer = OODScorer(config)
    previous_positions = None
    previous_forces = None
    previous_q = None
    for frame in iter_dump(
        Path("dump.lammpstrj"),
        type_map=config["lammps"]["type_map"],
        timestep_fs=config["lammps"]["timestep_fs"],
    ):
        if frame.forces is None:
            raise ValueError("LAMMPS dump must contain fx fy fz for this example.")
        q_values = smooth_coordination_numbers(frame.atoms)
        metrics = {
            "force": force_norms(frame.forces),
            "delta_force": delta_force_norms(frame.forces, previous_forces),
            "displacement": displacement_norms(
                frame.atoms.get_positions(),
                previous_positions,
                cell=frame.atoms.cell.array,
                pbc=frame.atoms.pbc,
            ),
            "rmin": minimum_neighbor_distances(frame.atoms),
            "delta_q": coordination_deltas(q_values, previous_q),
        }
        result = scorer.score_light(metrics, metadata={"backend": config["backend"]["mlip"]})
        print(frame.step, result.stage, result.max_score, result.trigger_reason)
        previous_positions = frame.atoms.get_positions().copy()
        previous_forces = frame.forces.copy()
        previous_q = q_values.copy()


if __name__ == "__main__":
    main()
