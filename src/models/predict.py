"""Interactive command-line inference for the selected classical model."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from src.models.inference import load_model_artifacts, predict_text


def parse_args() -> argparse.Namespace:
    """Parse optional model and text inputs for a prediction."""
    parser = argparse.ArgumentParser(description="Classify a news article as real or fake.")
    parser.add_argument(
        "--text",
        help="Article text to classify. Omit to paste text interactively.",
    )
    parser.add_argument(
        "--config",
        default="configs/classical.yaml",
        help="Path to the model configuration file.",
    )
    parser.add_argument(
        "--model",
        help="Artifact model to load; defaults to deployment.model_name in the config.",
    )
    return parser.parse_args()


def main() -> None:
    """Load the selected model and print one structured prediction."""
    args = parse_args()
    text = args.text or input("Paste a news headline or article: ").strip()
    vectorizer, classifier, config, metadata = load_model_artifacts(
        args.config,
        args.model,
    )
    prediction = predict_text(
        text,
        vectorizer,
        classifier,
        config.get("preprocessing"),
    )
    print(
        json.dumps(
            {
                "model": metadata["model_name"],
                **asdict(prediction),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
