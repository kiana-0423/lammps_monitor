# Test Suite

This directory contains all test assets for `hotspot_al`.

## Layout

- `unit/`: fast tests for module contracts and local data transforms.
- `e2e/`: offline end-to-end tests using fixtures and fake backends only.
- `integration/`: optional tests for real external programs.
- `fixtures/`: small LAMMPS, CP2K, Allegro, and structure fixtures.
- `fake_backends/`: deterministic fake Allegro, CP2K, and LAMMPS helpers.
- `scripts/`: repeatable test commands used locally and in CI.

## Default Offline Tests

Run the reproducible offline suite from the repository root:

```bash
bash tests/scripts/run_all_offline.sh
```

This runs bytecode compilation, recursive package imports, unit tests, and the
offline E2E pipeline. It does not require real Allegro, LAMMPS, or CP2K.

## External Integration Tests

Real external backends are opt-in:

```bash
RUN_EXTERNAL=1 LAMMPS_BIN=/path/to/lmp bash tests/scripts/run_integration_if_available.sh
RUN_EXTERNAL=1 CP2K_BIN=/path/to/cp2k.popt bash tests/scripts/run_integration_if_available.sh
RUN_EXTERNAL=1 ALLEGRO_MODEL=/path/to/model.pth bash tests/scripts/run_integration_if_available.sh
```

Backend-specific commands:

```bash
RUN_EXTERNAL=1 CP2K_BIN=cp2k.popt python -m pytest -q tests/integration -m cp2k
RUN_EXTERNAL=1 LAMMPS_BIN=lmp python -m pytest -q tests/integration -m lammps
```

If `RUN_EXTERNAL` is not `1`, or a backend-specific environment variable is
missing, the corresponding tests are skipped. CI does not install or run real
Allegro, LAMMPS, or CP2K.

Current real integration coverage:

- CP2K: generates a tiny H2 `ENERGY_FORCE` input, runs CP2K, and parses the real force output.
- LAMMPS: runs a tiny LJ `run 0`, writes a custom dump, and reads it through the project LAMMPS reader.
- Allegro: still limited to checking that `ALLEGRO_MODEL` points to a file.

Real Allegro inference, `pair_allegro`, and the Allegro-LAMMPS online loop are
next-stage integration work.

## Fake Backends

Fake backends validate internal data protocols without claiming physical
accuracy. They are used to check array shapes, generated files, and handoffs
between modules.

Passing offline tests means the internal Python protocol is coherent. It does
not mean the production Allegro-LAMMPS-CP2K active learning loop is complete.

## Fixtures

Fixtures are intentionally tiny and synthetic. They are suitable for parser,
writer, and data-contract tests only, not for scientific validation.
