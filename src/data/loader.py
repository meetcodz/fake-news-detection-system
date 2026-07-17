"""Dataset loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.data.clean import build_text_column, drop_invalid_rows
from utils.logging import get_logger

logger = get_logger(__name__)


def load_dataset(
    path: str | Path,
    text_column: str = "text",
    label_column: str = "label",
    title_column: str | None = None,
    combine_title_text: bool = False,
    drop_missing_text: bool = True,
    label_mapping: dict[Any, int] | None = None,
) -> tuple[list[str], list[int]]:
    """Load a CSV dataset and return parallel text and label lists.

    Args:
        path: Path to the CSV file.
        text_column: Column containing document text.
        label_column: Column containing binary labels (0=real, 1=fake).
        title_column: Optional title column to prepend to text.
        combine_title_text: Whether to join ``title_column`` with ``text_column``.
        drop_missing_text: Drop rows with empty text or missing labels.
        label_mapping: Optional mapping for non-binary label values.

    Returns:
        Tuple of (texts, labels).

    Raises:
        FileNotFoundError: If the dataset path does not exist.
        ValueError: If required columns are missing or data is invalid.
    """
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    frame = pd.read_csv(dataset_path)
    _validate_columns(frame, text_column, label_column, title_column)

    working = frame.copy()
    working["_document_text"] = build_text_column(
        working,
        text_column=text_column,
        title_column=title_column,
        combine_title_text=combine_title_text,
    )

    if drop_missing_text:
        working = drop_invalid_rows(working, "_document_text", label_column)

    if label_mapping:
        working[label_column] = working[label_column].map(label_mapping)

    _validate_binary_labels(working[label_column])

    texts = working["_document_text"].astype(str).tolist()
    labels = working[label_column].astype(int).tolist()

    if len(texts) == 0:
        raise ValueError(f"Dataset is empty after cleaning: {dataset_path}")

    logger.info(
        "Loaded dataset",
        extra={"path": str(dataset_path), "samples": len(texts)},
    )
    return texts, labels


def load_dataset_from_config(
    dataset_config: dict[str, Any],
    root: Path,
) -> tuple[list[str], list[int]]:
    """Load a dataset using paths and options from a config mapping."""
    dataset_path = Path(dataset_config["path"])
    if not dataset_path.is_absolute():
        dataset_path = root / dataset_path

    return load_dataset(
        dataset_path,
        text_column=dataset_config.get("text_column", "text"),
        label_column=dataset_config.get("label_column", "label"),
        title_column=dataset_config.get("title_column"),
        combine_title_text=dataset_config.get("combine_title_text", False),
        drop_missing_text=dataset_config.get("drop_missing_text", True),
        label_mapping=dataset_config.get("label_mapping"),
    )


def _validate_columns(
    frame: pd.DataFrame,
    text_column: str,
    label_column: str,
    title_column: str | None,
) -> None:
    """Ensure the dataframe contains the expected schema."""
    required = {text_column, label_column}
    if title_column:
        required.add(title_column)

    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")


def _validate_binary_labels(series: pd.Series) -> None:
    """Ensure labels are binary integers."""
    if series.isna().any():
        raise ValueError("Label column contains null values after cleaning")

    unique_labels = set(series.unique())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(
            f"Labels must be binary (0 or 1); found: {sorted(unique_labels)}"
        )
