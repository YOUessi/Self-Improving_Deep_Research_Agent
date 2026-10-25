"""Tests are deterministic and never require GPU/API or internet."""

from __future__ import annotations

from pathlib import Path

import pytest

from sidra.io import load_dataset


@pytest.fixture
def fixture_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "examples" / "fixture"


@pytest.fixture
def fixture_data(fixture_dir):
    return load_dataset(fixture_dir)
