"""Training and comparison pipeline for Stage 4 Transformer-based models."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

from src.models.evaluate import compute_binary_metrics
from src.models.pipeline import SplitData, load_split_data, resolve_project_path
from utils.config import get_project_root, load_config
from utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

_CHECKPOINT_JUNK = {"optimizer.pt", "scheduler.pt", "rng_state.pth"}


def _cleanup_training_checkpoints(checkpoint_dir: Path) -> None:
    """Delete large optimizer/scheduler/rng files that are useless after training."""
    if not checkpoint_dir.exists():
        return
    for path in checkpoint_dir.rglob("*"):
        if path.is_file() and path.name in _CHECKPOINT_JUNK:
            path.unlink(missing_ok=True)
            logger.info("Removed training checkpoint file: %s", path.name)


class TransformerDataset(Dataset):
    """PyTorch Dataset wrapping tokenizer encoding for sequence classification."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int] | None,
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def _numpy_softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax probabilities along the last axis in NumPy."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def compute_transformer_metrics(eval_pred: EvalPrediction) -> dict[str, Any]:
    """Compute standard binary metrics from transformer evaluation predictions."""
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    probs = _numpy_softmax(logits)[:, 1] if logits.ndim == 2 and logits.shape[1] == 2 else None
    return compute_binary_metrics(labels, preds, probs)


def train_transformer_classifier(
    config: dict[str, Any],
    root: Path,
    model_name: str,
    splits: SplitData,
) -> dict[str, Any]:
    """Fine-tune, evaluate, and save a HuggingFace transformer model."""
    output_cfg = config["output"]
    training_cfg = config["training"]
    
    # Path resolution
    model_dir = resolve_project_path(output_cfg["model_dir"], root) / model_name.replace("/", "_")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Initializing tokenizer and model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # Build datasets
    max_len = int(training_cfg.get("max_sequence_length", 256))
    train_dataset = TransformerDataset(splits.train_texts, splits.train_labels, tokenizer, max_len)
    val_dataset = TransformerDataset(splits.test_texts, splits.test_labels, tokenizer, max_len)
    
    # Resolve training device
    device_name = str(training_cfg.get("device", "auto")).lower()
    use_cpu = device_name == "cpu" or (device_name == "auto" and not torch.cuda.is_available())
    fp16 = bool(training_cfg.get("fp16", True)) and torch.cuda.is_available() and not use_cpu

    # Setup training arguments
    args = TrainingArguments(
        output_dir=str(model_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=float(training_cfg.get("learning_rate", 3e-5)),
        per_device_train_batch_size=int(training_cfg.get("batch_size", 16)),
        per_device_eval_batch_size=int(training_cfg.get("batch_size", 16)),
        num_train_epochs=int(training_cfg.get("epochs", 3)),
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),
        logging_steps=int(training_cfg.get("logging_steps", 100)),
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=fp16,
        use_cpu=use_cpu,
        report_to="none",
        warmup_ratio=float(training_cfg.get("warmup_ratio", 0.1)),
        seed=int(training_cfg.get("random_state", 42)),
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_transformer_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=int(training_cfg.get("patience", 1)))],
    )
    
    logger.info("Starting fine-tuning for model: %s", model_name)
    trainer.train()
    
    # Remove optimizer/scheduler/rng checkpoints immediately — they're large and
    # not needed after training completes.
    _cleanup_training_checkpoints(model_dir / "checkpoints")
    
    # Save best model and tokenizer
    trainer.save_model(str(model_dir))
    try:
        tokenizer.save_pretrained(str(model_dir))
    except OSError as exc:
        # Non-fatal: the tokenizer is already cached in the HuggingFace hub cache.
        logger.warning("Tokenizer save skipped (likely low disk): %s", exc)
    
    # Evaluate best model
    eval_results = trainer.evaluate()
    
    # Adapt metrics keys to fit classical structure
    metrics = {
        "accuracy": eval_results["eval_accuracy"],
        "precision": eval_results["eval_precision"],
        "recall": eval_results["eval_recall"],
        "f1": eval_results["eval_f1"],
        "roc_auc": eval_results.get("eval_roc_auc"),
    }
    
    # Save metadata
    metadata_path = model_dir / output_cfg.get("metadata_filename", "metadata.json")
    metadata = {
        "model_name": model_name,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": "cpu" if use_cpu else "cuda",
        "dataset": {
            "path": config["dataset"]["path"],
            "version": config["dataset"].get("version", "unspecified"),
            "training_samples": len(splits.train_labels),
            "test_samples": len(splits.test_labels),
        },
        "training": training_cfg,
        "metrics": metrics,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    
    # Save metrics.json
    metrics_path = model_dir / output_cfg.get("metrics_filename", "metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    
    return {
        "model_name": model_name,
        "metrics": metrics,
        "artifacts": {
            "model_dir": str(model_dir),
            "metadata": str(metadata_path),
            "metrics": str(metrics_path),
        },
    }


def compare_transformer_models(config_path: str | Path = "configs/transformer.yaml") -> pd.DataFrame:
    """Run Stage 4 comparison on transformer models."""
    root = get_project_root()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
        
    config = load_config(config_file)
    setup_logging(config)
    
    splits = load_split_data(config, root)
    
    # Subsampling logic for developer testing / sanity check runs
    max_samples = config["dataset"].get("max_samples")
    if max_samples is not None:
        max_samples = int(max_samples)
        logger.info("Applying max_samples subsampling: %d", max_samples)
        from sklearn.model_selection import train_test_split
        
        train_size = min(int(max_samples * 0.8), len(splits.train_texts))
        test_size = min(int(max_samples * 0.2), len(splits.test_texts))
        
        train_texts, _, train_labels, _ = train_test_split(
            splits.train_texts,
            splits.train_labels,
            train_size=train_size,
            random_state=42,
            stratify=splits.train_labels,
        )
        test_texts, _, test_labels, _ = train_test_split(
            splits.test_texts,
            splits.test_labels,
            train_size=test_size,
            random_state=42,
            stratify=splits.test_labels,
        )
        splits = SplitData(train_texts, test_texts, train_labels, test_labels)
        
    rows: list[dict[str, Any]] = []
    detailed_results: list[dict[str, Any]] = []
    
    for model_name in config["models"]:
        started_at = time.perf_counter()
        try:
            result = train_transformer_classifier(config, root, model_name, splits)
            duration_seconds = time.perf_counter() - started_at
            result["training_seconds"] = duration_seconds
            
            metrics = result["metrics"]
            rows.append({
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "training_seconds": duration_seconds,
            })
            detailed_results.append(result)
        except Exception as exc:
            logger.exception("Failed to train transformer model %s", model_name)
            
    if not rows:
        raise RuntimeError("No transformer models were trained successfully.")
        
    comparison = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    output_dir = resolve_project_path(config["output"]["model_dir"], root)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    comparison.to_csv(output_dir / config["output"]["comparison_csv"], index=False)
    (output_dir / config["output"]["comparison_json"]).write_text(json.dumps(detailed_results, indent=2), encoding="utf-8")
    
    logger.info("Transformer comparison complete. Models compared: %d", len(comparison))
    return comparison


def main() -> None:
    """CLI entrypoint to run transformer comparison."""
    parser = argparse.ArgumentParser(description="Stage 4 Transformer training and comparison.")
    parser.add_argument(
        "--config",
        default="configs/transformer.yaml",
        help="Path to the model configuration file.",
    )
    args = parser.parse_args()
    compare_transformer_models(args.config)


if __name__ == "__main__":
    main()
