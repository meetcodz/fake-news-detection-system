"""Tests for dataset row cleaning."""

from __future__ import annotations

import pandas as pd

from src.data.clean import build_text_column, drop_invalid_rows


def test_build_text_column_combines_title_and_text() -> None:
    frame = pd.DataFrame(
        {
            "title": ["Breaking news", ""],
            "text": ["Article body", "Only body"],
        }
    )

    combined = build_text_column(
        frame,
        text_column="text",
        title_column="title",
        combine_title_text=True,
    )

    assert combined.tolist() == ["Breaking news. Article body", "Only body"]


def test_drop_invalid_rows_removes_empty_text() -> None:
    frame = pd.DataFrame({"text": ["hello", "  ", "world"], "label": [0, 1, 1]})
    cleaned = drop_invalid_rows(frame, "text", "label")

    assert len(cleaned) == 2
    assert cleaned["text"].tolist() == ["hello", "world"]
