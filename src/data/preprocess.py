from __future__ import annotations

import re
from typing import Any

from utils.logging import get_logger

logger = get_logger(__name__)

_DATELINE_RE = re.compile(
    r"^[A-Z][A-Z\s/,\.]{1,40}\s*\([^)]{1,30}\)\s*[-–—]+\s*",
    re.MULTILINE,
)

_INLINE_SOURCE_RE = re.compile(r"\s*[|\-–]\ *(reuters|associated press|ap|afp|bbc|cnn|nyt|bloomberg)\b",
                               re.IGNORECASE)

def strip_datelines(text: str) -> str:

    text = _DATELINE_RE.sub("", text)
    text = _INLINE_SOURCE_RE.sub("", text)
    return text.strip()

def preprocess_text(text: str, config: dict[str, Any] | None = None) -> str:
                                                     
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}")

    settings = config or {}
    
    cleaned = text

    if settings.get("strip_datelines", True):
        cleaned = strip_datelines(cleaned)
    if settings.get("remove_urls", True):
        cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)
    if settings.get("remove_html", True):
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    if settings.get("lowercase", True):
        cleaned = cleaned.lower()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

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
                                                                                 
    processed, _ = preprocess_dataset(texts, labels=None, config=config)
    return processed

def preprocess_dataset(
    texts: list[str],
    labels: list[int] | None = None,
    config: dict[str, Any] | None = None,
) -> tuple[list[str], list[int]] | tuple[list[str], None]:
                                                                      
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
            logger.warning("Skipped document during preprocessing: %s", exc)

    if not processed_texts:
        raise ValueError("No valid documents remain after preprocessing")

    if skipped:
        logger.warning("Preprocessing skipped %d documents", skipped)

    if labels is None:
        return processed_texts, None
    return processed_texts, processed_labels
