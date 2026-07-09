# Contributing

This project is a research prototype for hotspot-localized active learning around
Allegro/LAMMPS/CP2K workflows. Keep changes small, testable, and compatible with
offline unit tests unless a change explicitly targets real external runtimes.

## Development Environment

Create a local virtual environment and install the editable package with dev
tools:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -e ".[dev]"
```

The default development install does not include Torch, CUDA, Allegro, NequIP,
LAMMPS, or CP2K. Install optional runtime extras only when working on those
paths.

## Checks

Run these before opening a pull request:

```bash
python -m ruff check src tests examples
python -m mypy
python -m pytest -q
```

External integration tests are skipped by default. Set `RUN_EXTERNAL=1` only in
an environment where the corresponding real runtime is available.

## Code Style

- Prefer existing local patterns and helper APIs.
- Keep public functions typed.
- Add focused tests for new behavior and regression fixes.
- Do not require real LAMMPS, CP2K, Allegro, or CUDA in unit tests.
- Use `ExtractedRegion` compatibility fields when adding extraction modes:
  `core_indices`, `inner_buffer_indices`, `outer_buffer_indices`,
  `boundary_indices`, and `h_cap_indices`.

## Adding Extraction Modes

Add a mode-specific strategy in `hotspot_al.active_learning.workflow` and keep
the extractor itself in `hotspot_al.extraction`. New extractors should return
`ExtractedRegion` and preserve original-to-local index mappings in metadata.

## Adding OOD Metrics

New metrics should provide per-atom arrays, be included in `OODScorer` weights,
and add trigger-reason coverage in tests. Keep expensive metrics lazy or staged
when possible.
