"""Integration tests for baseline training and inference."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.models.inference import load_baseline_artifacts, predict_text
from src.models.train import train_baseline
from utils.config import load_config


@pytest.fixture
def training_config_path(tmp_path, project_root) -> Path:
    """Write a temporary config that stores artifacts under tmp_path."""
    base_config = load_config(project_root / "configs/baseline.yaml")
    base_config["output"]["model_dir"] = str(tmp_path / "models")

    config_path = tmp_path / "baseline.yaml"
    import yaml

    config_path.write_text(yaml.dump(base_config), encoding="utf-8")
    return config_path


def test_train_baseline_produces_metrics_and_artifacts(
    training_config_path,
    project_root,
) -> None:
    result = train_baseline(training_config_path)

    assert result["metrics"]["f1"] >= 0.0
    assert Path(result["artifacts"]["vectorizer"]).exists()
    assert Path(result["artifacts"]["classifier"]).exists()

    metrics = json.loads(Path(result["artifacts"]["metrics"]).read_text(encoding="utf-8"))
    assert "accuracy" in metrics
    assert "confusion_matrix" in metrics


def test_predict_text_after_training(training_config_path) -> None:
    train_baseline(training_config_path)

    vectorizer, classifier, config = load_baseline_artifacts(training_config_path)
    result = predict_text(
        "Scientists confirm renewable energy breakthrough in new study.",
        vectorizer,
        classifier,
        config.get("preprocessing"),
    )

    assert result.label_name in {"real", "fake"}
    assert 0.0 <= result.fake_probability <= 1.0
