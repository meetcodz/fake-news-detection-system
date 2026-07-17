"""Model evaluation metrics for binary classification."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from utils.logging import get_logger

logger = get_logger(__name__)


def compute_binary_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    y_proba: list[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute standard binary classification metrics.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        y_proba: Optional predicted probabilities for the positive class.

    Returns:
        Dictionary of metric names to scalar values or nested structures.
    """
    labels = np.asarray(y_true)
    predictions = np.asarray(y_pred)

    if labels.shape != predictions.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    if len(labels) == 0:
        raise ValueError("Cannot compute metrics on an empty dataset")

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
    }

    if y_proba is not None:
        probabilities = np.asarray(y_proba)
        if len(np.unique(labels)) < 2:
            logger.warning("ROC AUC skipped: only one class present in y_true")
        else:
            metrics["roc_auc"] = float(roc_auc_score(labels, probabilities))

    logger.info(
        "Computed evaluation metrics",
        extra={"accuracy": metrics["accuracy"], "f1": metrics["f1"]},
    )
    return metrics
