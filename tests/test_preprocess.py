"""Tests for text preprocessing."""

from __future__ import annotations

import pytest

from src.data.preprocess import preprocess_dataset, preprocess_text


def test_preprocess_text_normalizes_urls_and_html() -> None:
    raw = "  Visit <b>https://example.com</b> for INFO  "
    cleaned = preprocess_text(raw)

    assert "http" not in cleaned
    assert "<b>" not in cleaned
    assert cleaned == "visit for info"


def test_preprocess_text_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        preprocess_text("   ", config={"min_text_length": 3})


def test_preprocess_dataset_keeps_labels_aligned() -> None:
    texts = ["Valid news article with enough length.", "   ", "Another valid article here."]
    labels = [0, 1, 1]

    processed_texts, processed_labels = preprocess_dataset(texts, labels)

    assert len(processed_texts) == len(processed_labels) == 2
    assert processed_labels == [0, 1]
