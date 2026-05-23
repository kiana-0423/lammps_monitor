#!/usr/bin/env bash
set -euo pipefail

if [[ "${RUN_EXTERNAL:-}" != "1" ]]; then
  echo "RUN_EXTERNAL is not 1; skipping integration tests."
  exit 0
fi

if [[ -n "${LAMMPS_BIN:-}" ]]; then
  python -m pytest -q tests/integration -m lammps
else
  echo "LAMMPS_BIN is not set; skipping LAMMPS integration tests."
fi

if [[ -n "${CP2K_BIN:-}" ]]; then
  python -m pytest -q tests/integration -m cp2k
else
  echo "CP2K_BIN is not set; skipping CP2K integration tests."
fi

if [[ -n "${ALLEGRO_MODEL:-}" ]]; then
  python -m pytest -q tests/integration -m allegro
else
  echo "ALLEGRO_MODEL is not set; skipping Allegro integration tests."
fi

