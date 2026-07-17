"""Text preprocessing for news classification."""

from __future__ import annotations

import re
from html import unescape
from typing import Any

from utils.logging import get_logger

logger = get_logger(__name__)

_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
_HTML_PATTERN = re.compile(r"<[^>]+>")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def preprocess_text(text: str, config: dict[str, Any] | None = None) -> str:
    """Clean and normalize a single text document.

    Args:
        text: Raw input text.
        config: Optional preprocessing configuration mapping.

    Returns:
        Cleaned text string.

    Raises:
        ValueError: If text is empty or below minimum length after cleaning.
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}")

    settings = config or {}
    cleaned = unescape(text.strip())

    if settings.get("remove_html", True):
        cleaned = _HTML_PATTERN.sub(" ", cleaned)

    if settings.get("remove_urls", True):
        cleaned = _URL_PATTERN.sub(" ", cleaned)

    if settings.get("lowercase", True):
        cleaned = cleaned.lower()

    cleaned = _WHITESPACE_PATTERN.sub(" ", cleaned).strip()

    min_length = int(settings.get("min_text_length", 1))
    if len(cleaned) < min_length:
        raise ValueError(
            f"Text too short after preprocessing (min {min_length} chars)"
        )

    return cleaned


def preprocess_corpus(
    texts: list[str],
    config: dict[str, Any] | None = None,
) -> list[str]:
    """Preprocess a batch of documents, skipping invalid entries with logging."""
    processed, _ = preprocess_dataset(texts, labels=None, config=config)
    return processed


def preprocess_dataset(
    texts: list[str],
    labels: list[int] | None = None,
    config: dict[str, Any] | None = None,
) -> tuple[list[str], list[int]] | tuple[list[str], None]:
    """Preprocess documents while keeping texts and labels aligned."""
    if labels is not None and len(texts) != len(labels):
        raise ValueError("texts and labels must have the same length")

    processed_texts: list[str] = []
    processed_labels: list[int] = []
    skipped = 0

    for index, text in enumerate(texts):
        try:
            processed_texts.append(preprocess_text(text, config))
            if labels is not None:
                processed_labels.append(labels[index])
        except ValueError as exc:
            skipped += 1
            logger.warning(
                "Skipped document during preprocessing",
                extra={"index": index, "reason": str(exc)},
            )

    if not processed_texts:
        raise ValueError("No valid documents remain after preprocessing")

    if skipped:
        logger.warning("Preprocessing skipped documents", extra={"count": skipped})

    if labels is None:
        return processed_texts, None
    return processed_texts, processed_labels
