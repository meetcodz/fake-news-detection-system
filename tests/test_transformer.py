"""Unit tests for Stage 4 Transformer-based models and pipelines."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from src.models.transformer import (
    TransformerDataset,
    _numpy_softmax,
    compute_transformer_metrics,
    train_transformer_classifier,
)
from src.models.pipeline import SplitData


def test_transformer_dataset() -> None:
    """Test that TransformerDataset yields properly shaped token sequences."""
    texts = ["First news article", "Second fake headline"]
    labels = [0, 1]
    
    # Mock tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": torch.ones((1, 10), dtype=torch.long),
        "attention_mask": torch.ones((1, 10), dtype=torch.long),
    }
    
    dataset = TransformerDataset(texts, labels, mock_tokenizer, max_length=10)
    
    assert len(dataset) == 2
    item = dataset[0]
    
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "labels" in item
    assert item["labels"].item() == 0
    
    # Check shape (squeeze should remove the batch dim)
    assert item["input_ids"].shape == (10,)
    assert item["attention_mask"].shape == (10,)
    mock_tokenizer.assert_called_with(
        "First news article",
        truncation=True,
        padding="max_length",
        max_length=10,
        return_tensors="pt",
    )


def test_numpy_softmax() -> None:
    """Test standard numpy softmax functionality."""
    logits = np.array([[1.0, 2.0], [3.0, 1.0]])
    probs = _numpy_softmax(logits)
    
    # Assert values sum to 1
    assert np.allclose(probs.sum(axis=-1), 1.0)
    # Check directionality
    assert probs[0, 1] > probs[0, 0]
    assert probs[1, 0] > probs[1, 1]


def test_compute_transformer_metrics() -> None:
    """Test metric computation from raw logits."""
    # Mock EvalPrediction
    eval_pred = MagicMock()
    # Logits: class 0 has higher logit for index 0, class 1 has higher for index 1
    eval_pred.predictions = np.array([[2.0, 0.5], [0.1, 1.5]])
    eval_pred.label_ids = np.array([0, 1])
    
    metrics = compute_transformer_metrics(eval_pred)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0


def test_train_transformer_classifier_dummy(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify transformer training pipeline setup on mock objects."""
    # Mock tokenizer and model
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    
    mock_trainer_instance = MagicMock()
    mock_trainer_instance.evaluate.return_value = {
        "eval_accuracy": 0.9,
        "eval_precision": 0.9,
        "eval_recall": 0.9,
        "eval_f1": 0.9,
    }
    
    monkeypatch.setattr(
        "src.models.transformer.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: mock_tokenizer,
    )
    monkeypatch.setattr(
        "src.models.transformer.AutoModelForSequenceClassification.from_pretrained",
        lambda *args, **kwargs: mock_model,
    )
    monkeypatch.setattr(
        "src.models.transformer.Trainer",
        lambda *args, **kwargs: mock_trainer_instance,
    )
    
    config = {
        "dataset": {
            "path": "data/raw/WELFake_Dataset.csv",
            "version": "WELFake_Dataset.csv",
        },
        "training": {
            "device": "cpu",
            "batch_size": 2,
            "epochs": 1,
            "learning_rate": 3e-5,
            "max_sequence_length": 10,
        },
        "output": {
            "model_dir": str(tmp_path / "models"),
            "metadata_filename": "metadata.json",
            "metrics_filename": "metrics.json",
        },
    }
    
    splits = SplitData(
        train_texts=["news text one", "news text two"],
        test_texts=["test news one", "test news two"],
        train_labels=[0, 1],
        test_labels=[0, 1],
    )
    
    result = train_transformer_classifier(
        config=config,
        root=project_root,
        model_name="mock-transformer-model",
        splits=splits,
    )
    
    assert result["model_name"] == "mock-transformer-model"
    assert result["metrics"]["accuracy"] == 0.9
    
    # Assert artifacts are written
    model_dir = tmp_path / "models" / "mock-transformer-model"
    assert (model_dir / "metadata.json").exists()
    assert (model_dir / "metrics.json").exists()
