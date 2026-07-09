"""Optional real Allegro/NequIP H2O model inference smoke test."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.allegro]


def test_allegro_h2o_inference_smoke() -> None:
    if os.environ.get("RUN_EXTERNAL") != "1":
        pytest.skip("Set RUN_EXTERNAL=1 to enable real Allegro integration tests.")

    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "smoke_allegro_h2o_inference.py"
    env = os.environ.copy()
    env.setdefault("ALLEGRO_SMOKE_DEVICE", "cpu")
    fixture_model = repo_root / "tests" / "fixtures" / "models" / "tiny_h2o_allegro_model.pth"
    if not env.get("ALLEGRO_MODEL_PATH") and fixture_model.is_file():
        env["ALLEGRO_MODEL_PATH"] = str(fixture_model)

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=repo_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )

    if (
        not os.environ.get("ALLEGRO_MODEL_PATH")
        and not fixture_model.is_file()
        and env.get("ALLEGRO_INFERENCE_REQUIRED", "0") != "1"
        and result.returncode == 0
    ):
        pytest.skip(result.stdout.strip())

    assert result.returncode == 0, (
        f"smoke_allegro_h2o_inference.py failed with exit code {result.returncode}\n"
        f"failure hint: {'model element mismatch' if result.returncode == 7 else 'see stdout/stderr'}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
