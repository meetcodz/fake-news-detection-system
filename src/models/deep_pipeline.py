"""GPU-aware training and comparison pipeline for Stage 3 sequence models."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from src.models.deep_learning import Vocabulary, build_deep_model, build_vocabulary, resolve_training_device
from src.models.evaluate import compute_binary_metrics
from src.models.pipeline import SplitData, load_split_data, resolve_project_path
from utils.logging import get_logger

logger = get_logger(__name__)


class TextSequenceDataset(Dataset[tuple[Tensor, Tensor]]):
    """Fixed-length token sequences paired with binary labels."""

    def __init__(self, texts: list[str], labels: list[int], vocabulary: Vocabulary, max_length: int) -> None:
        self.token_ids = torch.tensor([vocabulary.encode(text, max_length) for text in texts], dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.token_ids[index], self.labels[index]


@dataclass(frozen=True)
class DeepDataLoaders:
    """Train/test PyTorch data loaders fitted with one training vocabulary."""

    train: DataLoader[tuple[Tensor, Tensor]]
    test: DataLoader[tuple[Tensor, Tensor]]
    vocabulary: Vocabulary


def build_data_loaders(config: dict[str, Any], splits: SplitData) -> DeepDataLoaders:
    """Create reproducible data loaders and fit vocabulary on training text only."""
    vocabulary_config = config["vocabulary"]
    training_config = config["training"]
    vocabulary = build_vocabulary(splits.train_texts, vocabulary_config)
    max_length = int(vocabulary_config["max_sequence_length"])
    batch_size = int(training_config["batch_size"])
    workers = int(training_config.get("num_workers", 0))
    generator = torch.Generator().manual_seed(int(training_config.get("random_state", 42)))
    train_dataset = TextSequenceDataset(splits.train_texts, splits.train_labels, vocabulary, max_length)
    test_dataset = TextSequenceDataset(splits.test_texts, splits.test_labels, vocabulary, max_length)
    return DeepDataLoaders(
        train=DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, generator=generator),
        test=DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers),
        vocabulary=vocabulary,
    )


def train_deep_classifier(
    config: dict[str, Any],
    root: Path,
    model_name: str,
    model_config: dict[str, Any],
    loaders: DeepDataLoaders | None = None,
    splits: SplitData | None = None,
) -> dict[str, Any]:
    """Train, evaluate, and persist one GPU-backed Stage 3 model."""
    device = resolve_training_device(config["training"])
    splits = splits or load_split_data(config, root)
    loaders = loaders or build_data_loaders(config, splits)
    model = build_deep_model(model_name, loaders.vocabulary.size, model_config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )
    criterion = nn.CrossEntropyLoss()
    history, best_state, best_f1, stale_epochs = [], None, -1.0, 0
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        train_loss = _train_epoch(model, loaders.train, optimizer, criterion, device)
        metrics = _evaluate(model, loaders.test, device)
        history.append({"epoch": epoch, "train_loss": train_loss, **metrics})
        logger.info("Deep model epoch complete", extra={"model": model_name, "epoch": epoch, "f1": metrics["f1"]})
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(config["training"].get("patience", 2)):
                break
    if best_state is None:
        raise RuntimeError("Deep model training produced no checkpoint")
    model.load_state_dict(best_state)
    metrics = _evaluate(model, loaders.test, device)
    artifacts = _save_deep_artifacts(config, root, model_name, model, loaders.vocabulary, metrics, history, model_config, device, splits)
    return {"model_name": model_name, "metrics": metrics, "history": history, "artifacts": artifacts}


def _train_epoch(model: nn.Module, loader: DataLoader[tuple[Tensor, Tensor]], optimizer: AdamW, criterion: nn.Module, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for token_ids, labels in loader:
        optimizer.zero_grad()
        logits = model(token_ids.to(device))
        loss = criterion(logits, labels.to(device))
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * len(labels)
    return total_loss / len(loader.dataset)


def _evaluate(model: nn.Module, loader: DataLoader[tuple[Tensor, Tensor]], device: torch.device) -> dict[str, Any]:
    model.eval()
    labels, predictions, probabilities = [], [], []
    with torch.no_grad():
        for token_ids, batch_labels in loader:
            logits = model(token_ids.to(device))
            batch_probabilities = torch.softmax(logits, dim=1)[:, 1]
            probabilities.extend(batch_probabilities.cpu().tolist())
            predictions.extend(torch.argmax(logits, dim=1).cpu().tolist())
            labels.extend(batch_labels.tolist())
    return compute_binary_metrics(np.asarray(labels), np.asarray(predictions), np.asarray(probabilities))


def _save_deep_artifacts(config: dict[str, Any], root: Path, model_name: str, model: nn.Module, vocabulary: Vocabulary, metrics: dict[str, Any], history: list[dict[str, Any]], model_config: dict[str, Any], device: torch.device, splits: SplitData) -> dict[str, str]:
    output_config = config["output"]
    model_dir = resolve_project_path(output_config["model_dir"], root) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / output_config["checkpoint_filename"]
    vocabulary_path = model_dir / output_config["vocabulary_filename"]
    metrics_path = model_dir / output_config["metrics_filename"]
    metadata_path = model_dir / output_config["metadata_filename"]
    torch.save({"model_name": model_name, "model_config": model_config, "state_dict": model.cpu().state_dict()}, checkpoint_path)
    vocabulary_path.write_text(json.dumps(vocabulary.token_to_id), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    metadata_path.write_text(json.dumps({"model_name": model_name, "trained_at_utc": datetime.now(timezone.utc).isoformat(), "device": str(device), "dataset": {"path": config["dataset"]["path"], "version": config["dataset"].get("version", "unspecified"), "training_samples": len(splits.train_labels), "test_samples": len(splits.test_labels)}, "vocabulary": config["vocabulary"], "training": config["training"], "model_config": model_config, "metrics": metrics, "history": history}, indent=2), encoding="utf-8")
    return {"checkpoint": str(checkpoint_path), "vocabulary": str(vocabulary_path), "metrics": str(metrics_path), "metadata": str(metadata_path)}
