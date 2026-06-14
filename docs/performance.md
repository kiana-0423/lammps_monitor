# Performance Benchmarks

The monitor benchmark script compares direct atom-wise monitor helpers with the
shared-neighbor fast paths used by `OnlineMonitor`.

Run a quick local benchmark:

```bash
python tests/scripts/benchmark_monitor.py --atoms 100,500,1000 --repeats 3
```

Write CSV output for later comparison:

```bash
python tests/scripts/benchmark_monitor.py --atoms 100,500,1000,5000 --repeats 5 --output benchmark_monitor.csv
```

CSV columns:

- `metric`: monitor metric (`rmin`, `coordination`, `lj_residual`).
- `n_atoms`: number of atoms in the synthetic periodic cell.
- `direct_s` / `fast_s`: average wall time per call.
- `speedup`: `direct_s / fast_s`.
- `direct_peak_mb` / `fast_peak_mb`: peak Python allocation measured with
  `tracemalloc`.

The numbers are machine dependent and should be treated as regression data for
the current runtime, not as portable scientific benchmark results.
