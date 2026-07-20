"""Tests for portable notebook utilities."""

from __future__ import annotations

from pathlib import Path

from utils.notebook import find_project_root, setup_notebook_environment


def test_find_project_root_from_repo(project_root: Path) -> None:
    discovered = find_project_root()
    assert discovered == project_root


def test_setup_notebook_environment(project_root: Path, monkeypatch) -> None:
    import sys
    # Save sys.path copy
    original_path = list(sys.path)
    
    try:
        discovered = setup_notebook_environment(install_package=False)
        assert discovered == project_root
        assert str(project_root) in sys.path
    finally:
        sys.path = original_path

