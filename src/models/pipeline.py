"""Shared TF-IDF training pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
from sklearn.model_selection import train_test_split

from src.data.loader import load_dataset_from_config
from src.data.preprocess import preprocess_dataset
from src.features.tfidf import build_tfidf_vectorizer
from src.models.evaluate import compute_binary_metrics
from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class SplitData:
    """Train and test splits for text classification."""

    train_texts: list[str]
    test_texts: list[str]
    train_labels: list[int]
    test_labels: list[int]


@dataclass(frozen=True)
class TfidfFeatureSet:
    """Fitted TF-IDF features shared across comparable models."""

    vectorizer: Any
    x_train: Any
    x_test: Any
    train_labels: list[int]
    test_labels: list[int]


def resolve_project_path(path: str | Path, root: Path) -> Path:
    """Resolve a config path relative to the project root when needed."""
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = root / resolved
    return resolved


def load_split_data(config: dict[str, Any], root: Path) -> SplitData:
    """Load, preprocess, and split the dataset defined in config."""
    texts, labels = load_dataset_from_config(config["dataset"], root)
    processed_texts, processed_labels = preprocess_dataset(
        texts,
        labels,
        config.get("preprocessing"),
    )
    if config["training"].get("deduplicate", False):
        processed_texts, processed_labels = deduplicate_examples(
            processed_texts,
            processed_labels,
        )
    train_texts, test_texts, train_labels, test_labels = split_dataset(
        processed_texts,
        processed_labels,
        config["training"],
    )
    return SplitData(train_texts, test_texts, train_labels, test_labels)


def deduplicate_examples(texts: list[str], labels: list[int]) -> tuple[list[str], list[int]]:
    """Remove exact cleaned-text duplicates and ambiguous duplicate labels.

    This prevents identical documents from leaking across a random train/test split.
    Documents assigned conflicting labels are excluded because their ground truth is
    ambiguous.
    """
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have the same length")

    labels_by_text: dict[str, set[int]] = {}
    first_label_by_text: dict[str, int] = {}
    for text, label in zip(texts, labels, strict=True):
        labels_by_text.setdefault(text, set()).add(label)
        first_label_by_text.setdefault(text, label)

    unique_texts: list[str] = []
    unique_labels: list[int] = []
    for text, label in first_label_by_text.items():
        if len(labels_by_text[text]) == 1:
            unique_texts.append(text)
            unique_labels.append(label)

    removed = len(texts) - len(unique_texts)
    conflicts = sum(len(values) > 1 for values in labels_by_text.values())
    if removed:
        logger.info(
            "Removed duplicate documents before splitting",
            extra={"removed": removed, "conflicting_labels": conflicts},
        )
    if not unique_texts:
        raise ValueError("No documents remain after duplicate removal")
    return unique_texts, unique_labels


def split_dataset(
    texts: list[str],
    labels: list[int],
    training_config: dict[str, Any],
) -> tuple[list[str], list[str], list[int], list[int]]:
    """Split texts and labels into train and test sets."""
    test_size = float(training_config.get("test_size", 0.2))
    random_state = training_config.get("random_state", 42)
    stratify = labels if training_config.get("stratify", True) else None

    return train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


def build_tfidf_features(
    config: dict[str, Any],
    splits: SplitData,
) -> TfidfFeatureSet:
    """Fit a TF-IDF vectorizer once for model comparison."""
    vectorizer = build_tfidf_vectorizer(config["features"])
    x_train = vectorizer.fit_transform(splits.train_texts)
    x_test = vectorizer.transform(splits.test_texts)

    logger.info(
        "Built TF-IDF features",
        extra={
            "train_samples": len(splits.train_labels),
            "test_samples": len(splits.test_labels),
        },
    )
    return TfidfFeatureSet(
        vectorizer=vectorizer,
        x_train=x_train,
        x_test=x_test,
        train_labels=splits.train_labels,
        test_labels=splits.test_labels,
    )


def train_tfidf_classifier(
    config: dict[str, Any],
    root: Path,
    classifier: Any,
    model_name: str,
    output_dir: str | Path | None = None,
    save_artifacts: bool = True,
    splits: SplitData | None = None,
    features: TfidfFeatureSet | None = None,
    model_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train a classifier on TF-IDF features and optionally persist artifacts."""
    if features is None:
        dataset_splits = splits or load_split_data(config, root)
        features = build_tfidf_features(config, dataset_splits)

    classifier.fit(features.x_train, features.train_labels)
    predictions = classifier.predict(features.x_test).tolist()
    probabilities = _extract_positive_scores(classifier, features.x_test)

    metrics = compute_binary_metrics(
        features.test_labels,
        predictions,
        probabilities,
    )

    result: dict[str, Any] = {
        "model_name": model_name,
        "metrics": metrics,
        "artifacts": {},
    }

    if save_artifacts:
        result["artifacts"] = _save_artifacts(
            vectorizer=features.vectorizer,
            classifier=classifier,
            metrics=metrics,
            config=config,
            root=root,
            model_name=model_name,
            output_dir=output_dir,
            model_config=model_config,
            features=features,
        )

    logger.info(
        "Model training complete",
        extra={"model_name": model_name, "f1": metrics["f1"]},
    )
    return result


def _extract_positive_scores(classifier: Any, features: Any) -> list[float]:
    """Return positive-class scores for ROC AUC computation."""
    if hasattr(classifier, "predict_proba"):
        return classifier.predict_proba(features)[:, 1].tolist()
    if hasattr(classifier, "decision_function"):
        return classifier.decision_function(features).tolist()
    raise AttributeError(
        f"Classifier {type(classifier).__name__} lacks predict_proba/decision_function"
    )


def _save_artifacts(
    vectorizer: Any,
    classifier: Any,
    metrics: dict[str, Any],
    config: dict[str, Any],
    root: Path,
    model_name: str,
    output_dir: str | Path | None,
    model_config: dict[str, Any] | None,
    features: TfidfFeatureSet,
) -> dict[str, str]:
    """Persist model artifacts and evaluation metrics to disk."""
    output_cfg = config["output"]
    model_dir = Path(output_dir) if output_dir else Path(output_cfg["model_dir"])
    if not model_dir.is_absolute():
        model_dir = root / model_dir

    if model_name:
        model_dir = model_dir / model_name

    model_dir.mkdir(parents=True, exist_ok=True)

    vectorizer_path = model_dir / output_cfg.get("vectorizer_filename", "tfidf_vectorizer.joblib")
    classifier_path = model_dir / output_cfg.get(
        "classifier_filename",
        f"{model_name}.joblib",
    )
    metrics_path = model_dir / output_cfg.get("metrics_filename", "metrics.json")
    metadata_path = model_dir / output_cfg.get("metadata_filename", "metadata.json")

    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(classifier, classifier_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    metadata = {
        "model_name": model_name or config.get("model", {}).get("name", "baseline"),
        "classifier_class": type(classifier).__name__,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "path": config["dataset"]["path"],
            "version": config["dataset"].get("version", "unspecified"),
            "training_samples": len(features.train_labels),
            "test_samples": len(features.test_labels),
        },
        "preprocessing": config.get("preprocessing", {}),
        "features": config.get("features", {}),
        "training": config.get("training", {}),
        "model_config": model_config or config.get("model", {}),
        "metrics": metrics,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "vectorizer": str(vectorizer_path),
        "classifier": str(classifier_path),
        "metrics": str(metrics_path),
        "metadata": str(metadata_path),
    }
