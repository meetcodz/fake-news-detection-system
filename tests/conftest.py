from __future__ import annotations

from pathlib import Path

import pytest

from utils.config import get_project_root

@pytest.fixture
def project_root() -> Path:
                                          
    return get_project_root()

@pytest.fixture
def sample_texts() -> list[str]:
                                                          
    return [
        "Scientists publish peer-reviewed climate research findings.",
        "SHOCKING: One weird trick cures every disease instantly!",
        "The parliament passed a bipartisan infrastructure bill.",
        "Aliens control the weather and officials are hiding it!",
    ]

@pytest.fixture
def sample_labels() -> list[int]:
                                               
    return [0, 1, 0, 1]
