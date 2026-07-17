"""TF-IDF feature engineering."""

from __future__ import annotations

from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer

from utils.logging import get_logger

logger = get_logger(__name__)


def build_tfidf_vectorizer(config: dict[str, Any]) -> TfidfVectorizer:
    """Create a configured TF-IDF vectorizer from a feature config mapping.

    Args:
        config: Feature configuration (typically ``config['features']``).

    Returns:
        Unfitted ``TfidfVectorizer`` instance.
    """
    feature_type = config.get("type", "tfidf")
    if feature_type != "tfidf":
        raise ValueError(f"Unsupported feature type: {feature_type}")

    ngram_range = tuple(config.get("ngram_range", [1, 2]))

    vectorizer = TfidfVectorizer(
        max_features=config.get("max_features", 5000),
        ngram_range=ngram_range,
        min_df=config.get("min_df", 1),
        max_df=config.get("max_df", 1.0),
        sublinear_tf=config.get("sublinear_tf", True),
    )

    logger.info(
        "Built TF-IDF vectorizer",
        extra={
            "max_features": vectorizer.max_features,
            "ngram_range": ngram_range,
        },
    )
    return vectorizer
