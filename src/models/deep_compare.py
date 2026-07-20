"""CLI comparison runner for Stage 3 BiLSTM and GRU models."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.models.deep_pipeline import build_data_loaders, train_deep_classifier
from src.models.pipeline import load_split_data, resolve_project_path
from utils.config import get_project_root, load_config
from utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def compare_deep_models(config_path: str | Path = "configs/deep_learning.yaml") -> pd.DataFrame:
    """Train configured Stage 3 models on one shared split and vocabulary."""
    root = get_project_root()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config = load_config(config_file)
    setup_logging(config)
    _set_seed(int(config["training"].get("random_state", 42)))
    splits = load_split_data(config, root)
    loaders = build_data_loaders(config, splits)
    rows: list[dict[str, Any]] = []
    detailed_results: list[dict[str, Any]] = []
    for model_name in config["models"]:
        started_at = time.perf_counter()
        result = train_deep_classifier(
            config,
            root,
            model_name,
            config["model_configs"][model_name],
            loaders=loaders,
            splits=splits,
        )
        duration_seconds = time.perf_counter() - started_at
        result["training_seconds"] = duration_seconds
        metrics = result["metrics"]
        rows.append({"model": model_name, "accuracy": metrics["accuracy"], "precision": metrics["precision"], "recall": metrics["recall"], "f1": metrics["f1"], "roc_auc": metrics["roc_auc"], "training_seconds": duration_seconds})
        detailed_results.append(result)
    comparison = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    output_dir = resolve_project_path(config["output"]["model_dir"], root)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / config["output"]["comparison_csv"], index=False)
    (output_dir / config["output"]["comparison_json"]).write_text(json.dumps(detailed_results, indent=2), encoding="utf-8")
    logger.info("Deep model comparison complete", extra={"models": len(comparison)})
    return comparison


def _set_seed(seed: int) -> None:
    """Seed PyTorch randomness for repeatable Stage 3 experiments."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    """Run the Stage 3 comparison using the default configuration."""
    compare_deep_models()


if __name__ == "__main__":
    main()
