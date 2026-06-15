#!/usr/bin/env bash
set -euo pipefail

python -m compileall -q src tests examples
python tests/scripts/check_imports.py
python -m pytest -q tests/unit tests/e2e -m "not integration and not slow"

if [[ "${HOTSPOT_AL_CI:-}" == "1" ]]; then
  if [[ -n "${CP2K_BIN:-}" ]]; then
    RUN_EXTERNAL=1 python -m pytest -q tests/integration -m cp2k
  else
    echo "HOTSPOT_AL_CI=1 but CP2K_BIN is not set; skipping CP2K integration tests."
  fi
fi
