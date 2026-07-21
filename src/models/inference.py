"""Model-agnostic inference utilities for persisted TF-IDF classifiers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.preprocess import preprocess_text
from utils.config import get_project_root, load_config
from utils.logging import get_logger

logger = get_logger(__name__)

LABEL_MAP = {0: "real", 1: "fake"}


@dataclass(frozen=True)
class PredictionResult:
    """Structured probability-based inference output for one document."""

    label: int
    label_name: str
    fake_probability: float
    real_probability: float


def load_model_artifacts(
    config_path: str | Path = "configs/classical.yaml",
    model_name: str | None = None,
) -> tuple[TfidfVectorizer, Any, dict[str, Any], dict[str, Any]]:
    """Load a configured classical model, vectorizer, and training metadata.

    When ``model_name`` is omitted, the ``deployment.model_name`` configuration
    selects the serving model. Only classifiers exposing ``predict_proba`` can be
    served because credibility scores must be calibrated probabilities.
    """
    root = get_project_root()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file

    config = load_config(config_file)
    selected_model = model_name or config.get("deployment", {}).get("model_name")
    if not selected_model:
        raise ValueError("Specify model_name or configure deployment.model_name")

    output_cfg = config["output"]
    model_dir = Path(output_cfg["model_dir"])
    if not model_dir.is_absolute():
        model_dir = root / model_dir
    model_dir = model_dir / selected_model

    vectorizer_path = model_dir / output_cfg["vectorizer_filename"]
    classifier_path = model_dir / output_cfg["classifier_filename"]
    metadata_path = model_dir / output_cfg.get("metadata_filename", "metadata.json")
    missing = [path for path in (vectorizer_path, classifier_path, metadata_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Artifacts for model '{selected_model}' are missing: {', '.join(map(str, missing))}"
        )

    vectorizer = joblib.load(vectorizer_path)
    classifier = joblib.load(classifier_path)
    if not hasattr(classifier, "predict_proba"):
        raise TypeError(
            f"Model '{selected_model}' does not provide calibrated probabilities. "
            "Train it with calibration enabled before deployment."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return vectorizer, classifier, config, metadata


def load_baseline_artifacts(
    config_path: str | Path = "configs/baseline.yaml",
) -> tuple[TfidfVectorizer, Any, dict[str, Any]]:
    """Load baseline artifacts while preserving the original public API."""
    root = get_project_root()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config = load_config(config_file)
    output_cfg = config["output"]
    model_dir = Path(output_cfg["model_dir"])
    if not model_dir.is_absolute():
        model_dir = root / model_dir

    vectorizer_path = model_dir / output_cfg["vectorizer_filename"]
    classifier_path = model_dir / output_cfg["classifier_filename"]
    if not vectorizer_path.exists() or not classifier_path.exists():
        raise FileNotFoundError("Model artifacts not found. Train the baseline first via src.models.train")
    return joblib.load(vectorizer_path), joblib.load(classifier_path), config


def predict_text(
    text: str,
    vectorizer: TfidfVectorizer,
    classifier: Any,
    preprocessing_config: dict[str, Any] | None = None,
) -> PredictionResult:
    """Classify one document with any probability-enabled binary classifier."""
    if not hasattr(classifier, "predict_proba"):
        raise TypeError("Classifier must expose predict_proba for probability inference")
    cleaned = preprocess_text(text, preprocessing_config)
    features = vectorizer.transform([cleaned])
    label = int(classifier.predict(features)[0])
    probabilities = classifier.predict_proba(features)[0]
    result = PredictionResult(
        label=label,
        label_name=LABEL_MAP[label],
        fake_probability=float(probabilities[1]),
        real_probability=float(probabilities[0]),
    )
    logger.info("Prediction complete", extra={"label": result.label_name})
    return result


def predict_batch(
    texts: list[str],
    vectorizer: TfidfVectorizer,
    classifier: Any,
    preprocessing_config: dict[str, Any] | None = None,
) -> list[PredictionResult]:
    """Classify multiple documents with any probability-enabled classifier."""
    return [predict_text(text, vectorizer, classifier, preprocessing_config) for text in texts]


def load_deep_model_artifacts(
    config_path: str | Path = "configs/deep_learning.yaml",
    model_name: str | None = None,
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    """Load a configured deep learning model (BiLSTM/GRU), vocabulary, config, and metadata."""
    from src.models.deep_learning import Vocabulary, build_deep_model
    import torch

    root = get_project_root()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file

    config = load_config(config_file)
    selected_model = model_name or config.get("deployment", {}).get("model_name", "gru")

    output_cfg = config["output"]
    model_dir = Path(output_cfg["model_dir"])
    if not model_dir.is_absolute():
        model_dir = root / model_dir
    model_dir = model_dir / selected_model

    checkpoint_path = model_dir / output_cfg["checkpoint_filename"]
    vocabulary_path = model_dir / output_cfg["vocabulary_filename"]
    metadata_path = model_dir / output_cfg.get("metadata_filename", "metadata.json")

    missing = [path for path in (checkpoint_path, vocabulary_path, metadata_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Artifacts for deep model '{selected_model}' are missing: {', '.join(map(str, missing))}"
        )

    # Load vocabulary
    vocab_data = json.loads(vocabulary_path.read_text(encoding="utf-8"))
    vocabulary = Vocabulary(vocab_data)

    # Load metadata and checkpoint
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Reconstruct model
    model = build_deep_model(
        model_name=selected_model,
        vocabulary_size=vocabulary.size,
        config=checkpoint["model_config"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return vocabulary, model, config, metadata


def predict_deep_text(
    text: str,
    vocabulary: Any,
    model: Any,
    preprocessing_config: dict[str, Any] | None = None,
    max_sequence_length: int = 300,
) -> PredictionResult:
    """Classify one document using a deep learning sequence classifier."""
    import torch

    cleaned = preprocess_text(text, preprocessing_config)
    # Encode text
    token_ids = vocabulary.encode(cleaned, max_length=max_sequence_length)
    token_tensor = torch.tensor([token_ids], dtype=torch.long)

    with torch.no_grad():
        logits = model(token_tensor)
        probabilities = torch.softmax(logits, dim=1)[0].tolist()

    label = 1 if probabilities[1] >= 0.5 else 0
    result = PredictionResult(
        label=label,
        label_name=LABEL_MAP[label],
        fake_probability=float(probabilities[1]),
        real_probability=float(probabilities[0]),
    )
    logger.info("Deep prediction complete", extra={"label": result.label_name})
    return result

