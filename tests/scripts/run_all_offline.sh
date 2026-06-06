#!/usr/bin/env bash
set -euo pipefail

python -m compileall -q src tests examples
python tests/scripts/check_imports.py
python -m pytest -q tests/unit tests/e2e -m "not integration and not slow"
