"""Compare classical models using a shared TF-IDF pipeline."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.models.classical import SUPPORTED_CLASSICAL_MODELS, build_classical_model
from src.models.pipeline import (
    build_tfidf_features,
    load_split_data,
    resolve_project_path,
    train_tfidf_classifier,
)
from utils.config import get_project_root, load_config
from utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def compare_classical_models(config_path: str | Path) -> pd.DataFrame:
    """Train and evaluate all configured classical models on the same split.

    Returns:
        DataFrame sorted by F1 score (descending).
    """
    root = get_project_root()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file

    config = load_config(config_file)
    setup_logging(config)

    model_names = config.get("models", list(SUPPORTED_CLASSICAL_MODELS))
    model_configs = config.get("model_configs", {})
    default_model_config = config.get("model", {})
    output_dir = config["output"]["model_dir"]

    rows: list[dict[str, Any]] = []
    detailed_results: list[dict[str, Any]] = []
    splits = load_split_data(config, root)
    features = build_tfidf_features(config, splits)
    del splits
    gc.collect()

    for model_name in model_names:
        model_config = {
            **default_model_config,
            **model_configs.get(model_name, {}),
            "name": model_name,
        }
        classifier = build_classical_model(model_name, model_config)
        result = train_tfidf_classifier(
            config=config,
            root=root,
            classifier=classifier,
            model_name=model_name,
            output_dir=output_dir,
            save_artifacts=True,
            features=features,
            model_config=model_config,
        )

        metrics = result["metrics"]
        rows.append(
            {
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics.get("roc_auc"),
            }
        )
        detailed_results.append(result)
        del classifier
        gc.collect()

    comparison = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    _save_comparison(comparison, detailed_results, config, root)
    logger.info("Classical model comparison complete", extra={"models": len(comparison)})
    return comparison


def _save_comparison(
    comparison: pd.DataFrame,
    detailed_results: list[dict[str, Any]],
    config: dict[str, Any],
    root: Path,
) -> None:
    """Persist comparison table and detailed metrics."""
    output_dir = resolve_project_path(config["output"]["model_dir"], root)
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = output_dir / config["output"].get(
        "comparison_csv",
        "model_comparison.csv",
    )
    summary_path = output_dir / config["output"].get(
        "comparison_json",
        "model_comparison.json",
    )

    comparison.to_csv(comparison_path, index=False)
    summary_path.write_text(
        json.dumps(detailed_results, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    """CLI entry point for classical model comparison."""
    compare_classical_models("configs/classical.yaml")


if __name__ == "__main__":
    main()
