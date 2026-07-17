"""Dataset row cleaning helpers."""

from __future__ import annotations

import pandas as pd

from utils.logging import get_logger

logger = get_logger(__name__)


def build_text_column(
    frame: pd.DataFrame,
    text_column: str,
    title_column: str | None = None,
    combine_title_text: bool = False,
) -> pd.Series:
    """Build the document text column, optionally combining title and body."""
    text = frame[text_column].fillna("").astype(str).str.strip()

    if not title_column or title_column not in frame.columns:
        return text

    if not combine_title_text:
        return text

    title = frame[title_column].fillna("").astype(str).str.strip()
    combined = title.where(title == "", title + ". ") + text
    return combined.str.strip()


def drop_invalid_rows(
    frame: pd.DataFrame,
    text_column: str,
    label_column: str,
) -> pd.DataFrame:
    """Remove rows with empty text or missing labels."""
    cleaned = frame.copy()
    cleaned = cleaned[cleaned[label_column].notna()]
    cleaned = cleaned[cleaned[text_column].astype(str).str.strip() != ""]

    dropped = len(frame) - len(cleaned)
    if dropped:
        logger.warning("Dropped invalid dataset rows", extra={"count": dropped})

    return cleaned.reset_index(drop=True)
