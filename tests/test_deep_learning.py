"""Unit tests for Stage 3 vocabulary and recurrent model factories."""

from __future__ import annotations

import pytest
import torch

from src.models.deep_learning import build_deep_model, build_vocabulary, resolve_training_device


def test_build_vocabulary_encodes_unknown_tokens() -> None:
    vocabulary = build_vocabulary(["reliable news report", "news update"], {"max_size": 10, "min_frequency": 1})

    encoded = vocabulary.encode("unknown news", max_length=4)

    assert vocabulary.size >= 4
    assert encoded[0] == vocabulary.unknown_id
    assert encoded[1] == vocabulary.token_to_id["news"]
    assert encoded[-1] == vocabulary.pad_id


@pytest.mark.parametrize("model_name", ["bilstm", "gru"])
def test_deep_models_return_binary_logits(model_name: str) -> None:
    model = build_deep_model(model_name, 20, {"embedding_dim": 8, "hidden_dim": 4, "num_layers": 1, "dropout": 0.0, "bidirectional": True})

    logits = model(torch.tensor([[2, 3, 0], [4, 5, 6]], dtype=torch.long))

    assert logits.shape == (2, 2)


def test_cuda_requirement_fails_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        resolve_training_device({"device": "cuda"})


def test_train_deep_classifier_produces_artifacts(
    tmp_path,
    project_root,
) -> None:
    from src.models.deep_pipeline import train_deep_classifier

    # Define minimal config for CPU-based recurrent model training test
    config = {
        "dataset": {
            "path": str(project_root / "data/sample/sample_news.csv"),
            "text_column": "text",
            "label_column": "label",
            "drop_missing_text": True,
        },
        "preprocessing": {
            "lowercase": True,
            "remove_urls": True,
            "remove_html": True,
            "min_text_length": 10,
        },
        "training": {
            "test_size": 0.2,
            "random_state": 42,
            "stratify": True,
            "deduplicate": False,
            "device": "cpu",
            "batch_size": 2,
            "epochs": 1,
            "learning_rate": 0.01,
            "patience": 1,
            "num_workers": 0,
        },
        "vocabulary": {
            "max_size": 50,
            "min_frequency": 1,
            "max_sequence_length": 5,
        },
        "output": {
            "model_dir": str(tmp_path / "models"),
            "checkpoint_filename": "model.pt",
            "vocabulary_filename": "vocabulary.json",
            "metrics_filename": "metrics.json",
            "metadata_filename": "metadata.json",
        },
    }

    model_config = {
        "embedding_dim": 8,
        "hidden_dim": 4,
        "num_layers": 1,
        "dropout": 0.0,
        "bidirectional": True,
    }

    result = train_deep_classifier(
        config=config,
        root=project_root,
        model_name="bilstm",
        model_config=model_config,
    )

    assert result["model_name"] == "bilstm"
    assert "metrics" in result
    assert "history" in result
    assert "artifacts" in result

    # Check that output files exist
    model_dir = tmp_path / "models" / "bilstm"
    assert (model_dir / "model.pt").exists()
    assert (model_dir / "vocabulary.json").exists()
    assert (model_dir / "metrics.json").exists()
    assert (model_dir / "metadata.json").exists()


def test_deep_model_inference(project_root) -> None:
    """Verify that we can load a trained deep learning model and run predictions."""
    from src.models.inference import load_deep_model_artifacts, predict_deep_text

    try:
        vocabulary, model, config, metadata = load_deep_model_artifacts(
            config_path="configs/deep_learning.yaml",
            model_name="gru"
        )
    except FileNotFoundError:
        pytest.skip("Trained deep learning model 'gru' not found. Run deep_compare first.")

    res = predict_deep_text(
        text="BREAKING: Scandalous details about the politician revealed!",
        vocabulary=vocabulary,
        model=model,
        preprocessing_config=config.get("preprocessing"),
        max_sequence_length=config.get("vocabulary", {}).get("max_sequence_length", 300)
    )

    assert res.label in {0, 1}
    assert res.label_name in {"real", "fake"}
    assert 0.0 <= res.fake_probability <= 1.0
    assert 0.0 <= res.real_probability <= 1.0
    assert pytest.approx(res.fake_probability + res.real_probability) == 1.0


