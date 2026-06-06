"""Benchmark fast monitor neighbor paths against direct helpers."""

from __future__ import annotations

import time

import numpy as np
from ase import Atoms

from hotspot_al.monitor.geometry_monitor import minimum_neighbor_distances
from hotspot_al.monitor.geometry_monitor import minimum_neighbor_distances_fast
from hotspot_al.monitor.neighbor_utils import MonitorNeighbors


def _atoms(n_atoms: int) -> Atoms:
    rng = np.random.default_rng(7)
    positions = rng.random((n_atoms, 3)) * 40.0
    return Atoms("H" * n_atoms, positions=positions, cell=[40.0, 40.0, 40.0], pbc=True)


def _time_call(fn, repeats: int = 3) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start) / repeats


def main() -> None:
    print("n_atoms,direct_s,fast_s,speedup")
    for n_atoms in (100, 500, 1000):
        atoms = _atoms(n_atoms)
        neighbors = MonitorNeighbors(atoms, lj_cutoff=6.0, coordination_cutoff=6.0)
        direct = _time_call(lambda: minimum_neighbor_distances(atoms))
        fast = _time_call(lambda: minimum_neighbor_distances_fast(atoms, neighbors))
        speedup = direct / fast if fast > 0 else float("inf")
        print(f"{n_atoms},{direct:.6f},{fast:.6f},{speedup:.2f}")


if __name__ == "__main__":
    main()
