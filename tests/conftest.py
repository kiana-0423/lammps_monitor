"""Shared pytest helpers for the offline test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent

for path in (PROJECT_ROOT / "src", TESTS_DIR):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)


@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def fixtures_dir() -> Path:
    return TESTS_DIR / "fixtures"

