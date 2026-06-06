"""Benchmark fast monitor neighbor paths against direct helpers."""

from __future__ import annotations

import time

import numpy as np
from ase import Atoms

from hotspot_al.monitor.coordination_monitor import smooth_coordination_numbers, smooth_coordination_numbers_fast
from hotspot_al.monitor.geometry_monitor import minimum_neighbor_distances
from hotspot_al.monitor.geometry_monitor import minimum_neighbor_distances_fast
from hotspot_al.monitor.lj_residual import compute_lj_residuals
from hotspot_al.monitor.lj_residual_fast import compute_lj_residuals_fast
from hotspot_al.monitor.neighbor_utils import MonitorNeighbors


def _atoms(n_atoms: int) -> Atoms:
    rng = np.random.default_rng(7)
    positions = rng.random((n_atoms, 3)) * 40.0
    return Atoms("H" * n_atoms, positions=positions, cell=[40.0, 40.0, 40.0], pbc=True)


def _time_call(fn, repeats: int = 1) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start) / repeats


def main() -> None:
    print("metric,n_atoms,direct_s,fast_s,speedup")
    for n_atoms in (100, 500, 1000):
        atoms = _atoms(n_atoms)
        forces = np.ones((n_atoms, 3), dtype=float)
        neighbors = MonitorNeighbors(atoms, lj_cutoff=6.0, coordination_cutoff=6.0)
        rows = {
            "rmin": (
                _time_call(lambda: minimum_neighbor_distances(atoms)),
                _time_call(lambda: minimum_neighbor_distances_fast(atoms, neighbors)),
            ),
            "coordination": (
                _time_call(lambda: smooth_coordination_numbers(atoms)),
                _time_call(lambda: smooth_coordination_numbers_fast(atoms, neighbors)),
            ),
            "lj_residual": (
                _time_call(lambda: compute_lj_residuals(atoms, forces)),
                _time_call(lambda: compute_lj_residuals_fast(atoms, forces, neighbors)),
            ),
        }
        for metric, (direct, fast) in rows.items():
            speedup = direct / fast if fast > 0 else float("inf")
            print(f"{metric},{n_atoms},{direct:.6f},{fast:.6f},{speedup:.2f}")


if __name__ == "__main__":
    main()
