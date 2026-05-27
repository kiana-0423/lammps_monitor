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

Allegro/NequIP macOS smoke tests:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate allegro-mac

# Stage 1: Allegro runtime smoke test
python scripts/smoke_allegro.py

RUN_EXTERNAL=1 ALLEGRO_SMOKE_DEVICE=cpu pytest tests/integration/test_allegro_real.py -q
RUN_EXTERNAL=1 ALLEGRO_SMOKE_DEVICE=auto pytest tests/integration/test_allegro_real.py -q
RUN_EXTERNAL=1 ALLEGRO_SMOKE_DEVICE=mps pytest tests/integration/test_allegro_real.py -q

# Stage 2: H2O model inference smoke test
RUN_EXTERNAL=1 pytest tests/integration/test_allegro_h2o_inference_real.py -q

python scripts/make_tiny_h2o_dataset.py
python scripts/train_tiny_allegro_h2o.py

RUN_EXTERNAL=1 \
ALLEGRO_SMOKE_DEVICE=cpu \
ALLEGRO_MODEL_PATH=/path/to/h_o_model.pth \
ALLEGRO_INFERENCE_REQUIRED=1 \
pytest tests/integration/test_allegro_h2o_inference_real.py -q

ALLEGRO_SMOKE_DEVICE=cpu \
ALLEGRO_MODEL_PATH=/path/to/h_o_model.pth \
ALLEGRO_INFERENCE_REQUIRED=1 \
python scripts/smoke_allegro_h2o_inference.py
```

`ALLEGRO_SMOKE_DEVICE=auto` reports MPS availability but runs the stable smoke
path on CPU. The MPS command is an experimental tensor check and should be run
only when PyTorch reports MPS as available.

The H2O inference smoke test requires an existing Allegro/NequIP model that
supports H and O elements. If `ALLEGRO_MODEL_PATH` is not set but
`tests/fixtures/models/tiny_h2o_allegro_model.pth` exists, the pytest test uses
that local tiny fixture. If the model does not support H/O, the result is a
model element mismatch, not an Allegro/NequIP environment failure.

`scripts/make_tiny_h2o_dataset.py` writes a tiny synthetic H2O-like extxyz file
with toy energy/forces. `scripts/train_tiny_allegro_h2o.py` trains a tiny
NequIP fallback checkpoint for smoke-test inference when Allegro training is
not stable in the installed API. The dataset and model are synthetic smoke-test
fixtures only: they are not DFT data and have no physical or chemical accuracy.
No external model is downloaded automatically. CUDA is outside the scope of
these macOS smoke tests. LAMMPS `pair_allegro` coupling and CP2K smoke tests are
next-stage checks and are not covered here.

If `RUN_EXTERNAL` is not `1`, or a backend-specific environment variable is
missing, the corresponding tests are skipped. CI does not install or run real
Allegro, LAMMPS, or CP2K.

Current real integration coverage:

- CP2K: generates a tiny H2 `ENERGY_FORCE` input, runs CP2K, and parses the real force output.
- LAMMPS: runs a tiny LJ `run 0`, writes a custom dump, and reads it through the project LAMMPS reader.
- Allegro Stage 1: imports `torch`, `nequip`, `allegro`, and `ase`, reports
  CPU/MPS status, and runs tiny tensor sanity checks without training or large
  datasets.
- Allegro Stage 2: optionally loads an existing Allegro/NequIP model and runs a
  single H2O ASE calculator inference when `ALLEGRO_MODEL_PATH` is provided.

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
