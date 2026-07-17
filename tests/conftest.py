"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from utils.config import get_project_root


@pytest.fixture
def project_root() -> Path:
    """Return the repository root path."""
    return get_project_root()


@pytest.fixture
def sample_texts() -> list[str]:
    """Small labeled-style text samples for unit tests."""
    return [
        "Scientists publish peer-reviewed climate research findings.",
        "SHOCKING: One weird trick cures every disease instantly!",
        "The parliament passed a bipartisan infrastructure bill.",
        "Aliens control the weather and officials are hiding it!",
    ]


@pytest.fixture
def sample_labels() -> list[int]:
    """Labels aligned with ``sample_texts``."""
    return [0, 1, 0, 1]
