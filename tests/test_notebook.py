"""Tests for portable notebook utilities."""

from __future__ import annotations

from pathlib import Path

from utils.notebook import find_project_root


def test_find_project_root_from_repo(project_root: Path) -> None:
    discovered = find_project_root()
    assert discovered == project_root
