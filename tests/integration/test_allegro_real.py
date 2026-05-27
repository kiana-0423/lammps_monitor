"""Optional real Allegro/NequIP runtime smoke test."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = [pytest.mark.integration, pytest.mark.allegro]


def test_allegro_runtime_smoke() -> None:
    if os.environ.get("RUN_EXTERNAL") != "1":
        pytest.skip("Set RUN_EXTERNAL=1 to enable real Allegro integration tests.")

    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "smoke_allegro.py"
    env = os.environ.copy()
    env.setdefault("ALLEGRO_SMOKE_DEVICE", "auto")

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=repo_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )

    assert result.returncode == 0, (
        f"smoke_allegro.py failed with exit code {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
