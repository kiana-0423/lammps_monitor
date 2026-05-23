"""Optional real Allegro runtime smoke test."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


pytestmark = [pytest.mark.integration, pytest.mark.allegro]


def test_allegro_model_path_is_available() -> None:
    if os.environ.get("RUN_EXTERNAL") != "1":
        pytest.skip("Set RUN_EXTERNAL=1 to enable real Allegro integration tests.")
    model_path = os.environ.get("ALLEGRO_MODEL")
    if not model_path:
        pytest.skip("ALLEGRO_MODEL is not set.")

    assert Path(model_path).is_file()
