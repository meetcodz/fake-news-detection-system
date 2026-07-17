"""Tests for dataset loading."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.loader import load_dataset


def test_load_dataset_reads_sample_csv(project_root) -> None:
    texts, labels = load_dataset(project_root / "data/sample/sample_news.csv")

    assert len(texts) == len(labels) == 15
    assert set(labels) == {0, 1}


def test_load_dataset_combines_title_and_text(tmp_path) -> None:
    csv_path = tmp_path / "news.csv"
    pd.DataFrame(
        {
            "title": ["Headline", None],
            "text": ["Body text here", "Only text"],
            "label": [0, 1],
        }
    ).to_csv(csv_path, index=False)

    texts, labels = load_dataset(
        csv_path,
        title_column="title",
        combine_title_text=True,
    )

    assert texts == ["Headline. Body text here", "Only text"]
    assert labels == [0, 1]


@pytest.mark.slow
def test_load_welfake_dataset(project_root) -> None:
    dataset_path = project_root / "data/raw/WELFake_Dataset.csv"
    if not dataset_path.exists():
        return

    texts, labels = load_dataset(
        dataset_path,
        text_column="text",
        title_column="title",
        combine_title_text=True,
        drop_missing_text=True,
    )

    assert len(texts) == len(labels)
    assert len(texts) > 70000
    assert set(labels) == {0, 1}


def test_load_dataset_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset("data/missing.csv")


def test_load_dataset_missing_column_raises(tmp_path) -> None:
    csv_path = tmp_path / "bad.csv"
    pd.DataFrame({"content": ["hello"], "label": [0]}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_dataset(csv_path, text_column="text", label_column="label")


def test_load_dataset_rejects_non_binary_labels(tmp_path) -> None:
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame({"text": ["hello"], "label": [2]}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="binary"):
        load_dataset(csv_path)
