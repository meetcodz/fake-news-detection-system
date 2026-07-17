"""Tests for TF-IDF feature engineering."""

from __future__ import annotations

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from src.features.tfidf import build_tfidf_vectorizer


def test_build_tfidf_vectorizer_returns_configured_instance() -> None:
    config = {
        "type": "tfidf",
        "max_features": 100,
        "ngram_range": [1, 2],
        "min_df": 1,
        "max_df": 0.9,
        "sublinear_tf": True,
    }

    vectorizer = build_tfidf_vectorizer(config)

    assert isinstance(vectorizer, TfidfVectorizer)
    assert vectorizer.max_features == 100
    assert vectorizer.ngram_range == (1, 2)


def test_build_tfidf_vectorizer_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="Unsupported feature type"):
        build_tfidf_vectorizer({"type": "bow"})
