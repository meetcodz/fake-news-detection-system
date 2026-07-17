"""Tests for evaluation metrics."""

from __future__ import annotations

import pytest

from src.models.evaluate import compute_binary_metrics


def test_compute_binary_metrics_returns_expected_keys() -> None:
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 0]
    y_proba = [0.1, 0.6, 0.8, 0.4]

    metrics = compute_binary_metrics(y_true, y_pred, y_proba)

    assert metrics["accuracy"] == 0.5
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]


def test_compute_binary_metrics_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        compute_binary_metrics([0, 1], [0])
