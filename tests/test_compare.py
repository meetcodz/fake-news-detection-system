"""Tests for classical model comparison."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from src.models.classical import build_classical_model
from src.models.compare import compare_classical_models


def test_build_classical_model_supports_core_estimators() -> None:
    logistic = build_classical_model("logistic_regression", {"C": 1.0})
    naive_bayes = build_classical_model("naive_bayes", {"alpha": 0.5})

    assert logistic.__class__.__name__ == "LogisticRegression"
    assert naive_bayes.__class__.__name__ == "MultinomialNB"


def test_compare_classical_models_writes_comparison_table(
    tmp_path,
    project_root,
) -> None:
    base_config_path = project_root / "configs/classical.yaml"
    base_config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    base_config["models"] = ["logistic_regression", "naive_bayes"]
    base_config["output"]["model_dir"] = str(tmp_path / "models")

    config_path = tmp_path / "classical.yaml"
    config_path.write_text(yaml.dump(base_config), encoding="utf-8")

    comparison = compare_classical_models(config_path)

    assert isinstance(comparison, pd.DataFrame)
    assert len(comparison) == 2
    assert {"model", "accuracy", "precision", "recall", "f1"}.issubset(comparison.columns)

    csv_path = tmp_path / "models" / "model_comparison.csv"
    json_path = tmp_path / "models" / "model_comparison.json"
    assert csv_path.exists()
    assert json_path.exists()
    assert len(json.loads(json_path.read_text(encoding="utf-8"))) == 2
