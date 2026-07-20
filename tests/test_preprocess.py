"""Tests for text preprocessing."""

from __future__ import annotations

import pytest

from src.data.preprocess import preprocess_dataset, preprocess_text, strip_datelines


def test_preprocess_text_normalizes_urls_and_html() -> None:
    raw = "  Visit <b>https://example.com</b> for INFO  "
    cleaned = preprocess_text(raw)

    assert "http" not in cleaned
    assert "<b>" not in cleaned
    assert cleaned == "visit for info"


def test_preprocess_text_respects_cleanup_flags() -> None:
    """Configured cleanup operations can be disabled when needed."""
    raw = "Visit <b>HTTPS://EXAMPLE.COM</b>"

    cleaned = preprocess_text(
        raw,
        {
            "lowercase": False,
            "remove_urls": False,
            "remove_html": False,
            "strip_datelines": False,
        },
    )

    assert cleaned == raw


def test_preprocess_text_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        preprocess_text("   ", config={"min_text_length": 3})


def test_preprocess_dataset_keeps_labels_aligned() -> None:
    texts = ["Valid news article with enough length.", "   ", "Another valid article here."]
    labels = [0, 1, 1]

    processed_texts, processed_labels = preprocess_dataset(texts, labels)

    assert len(processed_texts) == len(processed_labels) == 2
    assert processed_labels == [0, 1]


def test_strip_datelines_removes_reuters_prefix() -> None:
    """Wire-service datelines are stripped so the model cannot learn source fingerprints."""
    reuters = "WASHINGTON (Reuters) - The senate voted on the bill today."
    assert strip_datelines(reuters) == "The senate voted on the bill today."


def test_strip_datelines_removes_ap_prefix() -> None:
    ap = "NEW YORK (AP) - Apple reported strong quarterly earnings."
    assert strip_datelines(ap) == "Apple reported strong quarterly earnings."


def test_strip_datelines_leaves_normal_text_unchanged() -> None:
    normal = "Scientists publish peer-reviewed findings on climate change."
    assert strip_datelines(normal) == normal


def test_preprocess_text_strips_datelines_by_default() -> None:
    """Dateline stripping is applied automatically unless explicitly disabled."""
    raw = "LONDON (Reuters) - British Prime Minister called an emergency session."
    cleaned = preprocess_text(raw)
    assert "reuters" not in cleaned
    assert "london" not in cleaned
    assert "british prime minister" in cleaned
