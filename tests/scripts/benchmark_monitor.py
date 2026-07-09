"""Benchmark fast monitor neighbor paths against direct helpers."""

from __future__ import annotations

import argparse
import csv
import json
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

from hotspot_al.monitor.coordination_monitor import (  # noqa: E402
    smooth_coordination_numbers,
    smooth_coordination_numbers_fast,
)
from hotspot_al.monitor.geometry_monitor import (  # noqa: E402
    minimum_neighbor_distances,
    minimum_neighbor_distances_fast,
)
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


def _baseline_key(row: dict[str, object]) -> str:
    return f"{row['metric']}:{row['n_atoms']}"


def _as_float(value: object) -> float:
    if isinstance(value, str | int | float):
        return float(value)
    msg = f"Expected numeric benchmark value, got {type(value).__name__}"
    raise TypeError(msg)


def _load_baseline(path: Path) -> dict[str, dict[str, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("results", data if isinstance(data, list) else [])
    return {
        _baseline_key(row): {"fast_s": _as_float(row["fast_s"]), "direct_s": _as_float(row["direct_s"])}
        for row in rows
    }


def _write_json(path: Path, rows: list[dict[str, object]], *, threshold: float) -> None:
    payload = {"threshold": threshold, "results": rows}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _regressions(
    rows: list[dict[str, object]],
    baseline: dict[str, dict[str, float]],
    *,
    threshold: float,
) -> list[str]:
    failures: list[str] = []
    for row in rows:
        key = _baseline_key(row)
        if key not in baseline:
            continue
        current_fast = _as_float(row["fast_s"])
        baseline_fast = baseline[key]["fast_s"]
        if baseline_fast > 0.0 and current_fast > baseline_fast * (1.0 + threshold):
            failures.append(f"{key} fast_s {current_fast:.6f}s > baseline {baseline_fast:.6f}s by more than {threshold:.0%}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atoms", default="100,500,1000", help="Comma-separated atom counts to benchmark.")
    parser.add_argument("--repeats", type=int, default=1, help="Repeats per metric and atom count.")
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV output path.")
    parser.add_argument("--json-output", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--baseline", type=Path, default=None, help="Optional JSON baseline for regression checks.")
    parser.add_argument("--threshold", type=float, default=0.2, help="Allowed performance regression fraction.")
    parser.add_argument("--warn-only", action="store_true", help="Print regressions without failing.")
    args = parser.parse_args()

    output_handle = args.output.open("w", newline="", encoding="utf-8") if args.output is not None else sys.stdout
    writer = csv.DictWriter(
        output_handle,
        fieldnames=["metric", "n_atoms", "direct_s", "fast_s", "speedup", "direct_peak_mb", "fast_peak_mb"],
    )
    writer.writeheader()
    result_rows: list[dict[str, object]] = []
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
            row = {
                "metric": metric,
                "n_atoms": n_atoms,
                "direct_s": round(direct_s, 6),
                "fast_s": round(fast_s, 6),
                "speedup": round(speedup, 2),
                "direct_peak_mb": round(direct_peak_mb, 3),
                "fast_peak_mb": round(fast_peak_mb, 3),
            }
            result_rows.append(row)
            writer.writerow(row)
    if args.output is not None:
        output_handle.close()
    if args.json_output is not None:
        _write_json(args.json_output, result_rows, threshold=args.threshold)
    if args.baseline is not None:
        failures = _regressions(result_rows, _load_baseline(args.baseline), threshold=args.threshold)
        for failure in failures:
            print(f"PERF REGRESSION: {failure}", file=sys.stderr)
        if failures and not args.warn_only:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
