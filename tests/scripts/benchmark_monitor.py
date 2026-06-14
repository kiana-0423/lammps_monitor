"""Benchmark fast monitor neighbor paths against direct helpers."""

from __future__ import annotations

import argparse
import csv
import sys
import time
import tracemalloc
from collections.abc import Callable
from pathlib import Path

import numpy as np
from ase import Atoms

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hotspot_al.monitor.coordination_monitor import smooth_coordination_numbers, smooth_coordination_numbers_fast  # noqa: E402
from hotspot_al.monitor.geometry_monitor import minimum_neighbor_distances  # noqa: E402
from hotspot_al.monitor.geometry_monitor import minimum_neighbor_distances_fast  # noqa: E402
from hotspot_al.monitor.lj_residual import compute_lj_residuals  # noqa: E402
from hotspot_al.monitor.lj_residual_fast import compute_lj_residuals_fast  # noqa: E402
from hotspot_al.monitor.neighbor_utils import MonitorNeighbors  # noqa: E402


def _atoms(n_atoms: int) -> Atoms:
    rng = np.random.default_rng(7)
    positions = rng.random((n_atoms, 3)) * 40.0
    return Atoms("H" * n_atoms, positions=positions, cell=[40.0, 40.0, 40.0], pbc=True)


def _measure_call(fn: Callable[[], object], repeats: int = 1) -> tuple[float, float]:
    peak_bytes = 0
    start = time.perf_counter()
    for _ in range(repeats):
        tracemalloc.start()
        fn()
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_bytes = max(peak_bytes, peak)
    return (time.perf_counter() - start) / repeats, peak_bytes / (1024 * 1024)


def _parse_atoms(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atoms", default="100,500,1000", help="Comma-separated atom counts to benchmark.")
    parser.add_argument("--repeats", type=int, default=1, help="Repeats per metric and atom count.")
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV output path.")
    args = parser.parse_args()

    output_handle = args.output.open("w", newline="", encoding="utf-8") if args.output is not None else sys.stdout
    writer = csv.DictWriter(
        output_handle,
        fieldnames=["metric", "n_atoms", "direct_s", "fast_s", "speedup", "direct_peak_mb", "fast_peak_mb"],
    )
    writer.writeheader()
    for n_atoms in _parse_atoms(args.atoms):
        atoms = _atoms(n_atoms)
        forces = np.ones((n_atoms, 3), dtype=float)
        neighbors = MonitorNeighbors(atoms, lj_cutoff=6.0, coordination_cutoff=6.0)
        rows = {
            "rmin": (
                lambda: minimum_neighbor_distances(atoms),
                lambda: minimum_neighbor_distances_fast(atoms, neighbors),
            ),
            "coordination": (
                lambda: smooth_coordination_numbers(atoms),
                lambda: smooth_coordination_numbers_fast(atoms, neighbors),
            ),
            "lj_residual": (
                lambda: compute_lj_residuals(atoms, forces),
                lambda: compute_lj_residuals_fast(atoms, forces, neighbors),
            ),
        }
        for metric, (direct, fast) in rows.items():
            direct_s, direct_peak_mb = _measure_call(direct, repeats=max(1, args.repeats))
            fast_s, fast_peak_mb = _measure_call(fast, repeats=max(1, args.repeats))
            speedup = direct_s / fast_s if fast_s > 0 else float("inf")
            writer.writerow(
                {
                    "metric": metric,
                    "n_atoms": n_atoms,
                    "direct_s": f"{direct_s:.6f}",
                    "fast_s": f"{fast_s:.6f}",
                    "speedup": f"{speedup:.2f}",
                    "direct_peak_mb": f"{direct_peak_mb:.3f}",
                    "fast_peak_mb": f"{fast_peak_mb:.3f}",
                }
            )
    if args.output is not None:
        output_handle.close()


if __name__ == "__main__":
    main()
