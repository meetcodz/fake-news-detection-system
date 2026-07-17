"""Baseline model training orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.models.baseline import build_logistic_regression
from src.models.pipeline import train_tfidf_classifier
from utils.config import get_project_root, load_config
from utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def train_baseline(config_path: str | Path) -> dict[str, Any]:
    """Train a TF-IDF + logistic regression baseline and persist artifacts."""
    root = get_project_root()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file

    config = load_config(config_file)
    setup_logging(config)

    classifier = build_logistic_regression(config["model"])
    result = train_tfidf_classifier(
        config=config,
        root=root,
        classifier=classifier,
        model_name="",
        output_dir=config["output"]["model_dir"],
        save_artifacts=True,
        model_config=config["model"],
    )

    logger.info("Baseline training complete", extra={"f1": result["metrics"]["f1"]})
    return result


def main() -> None:
    """CLI entry point for baseline training."""
    train_baseline("configs/baseline.yaml")


if __name__ == "__main__":
    main()
