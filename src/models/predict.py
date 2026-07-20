"""Interactive command-line inference with dual-model routing.

Short inputs (< 200 chars) are automatically routed to the headline-tuned SVM.
Longer inputs go to the full article model.
Borderline predictions (35-65% fake probability) are surfaced as 'uncertain'.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from src.models.inference import load_model_artifacts, predict_text

_HEADLINE_THRESHOLD_CHARS = 200


def parse_args() -> argparse.Namespace:
    """Parse optional overrides for config paths and article text."""
    parser = argparse.ArgumentParser(
        description="Classify a news headline or article as real, fake, or uncertain."
    )
    parser.add_argument(
        "--text",
        help="Text to classify. Omit to enter text interactively.",
    )
    parser.add_argument(
        "--article-config",
        default="configs/classical.yaml",
        dest="article_config",
        help="Config for the full-article model (default: configs/classical.yaml).",
    )
    parser.add_argument(
        "--headline-config",
        default="configs/headline.yaml",
        dest="headline_config",
        help="Config for the headline model (default: configs/headline.yaml).",
    )
    return parser.parse_args()


def _apply_threshold(fake_prob: float, deployment_cfg: dict) -> tuple[int, str]:
    """Apply configurable threshold + uncertain band to a raw fake probability."""
    threshold = float(deployment_cfg.get("fake_threshold", 0.50))
    band = deployment_cfg.get("uncertain_band", [])
    if band and len(band) == 2:
        lo, hi = float(band[0]), float(band[1])
        if lo < fake_prob < hi:
            return -1, "uncertain"
    return (1, "fake") if fake_prob >= threshold else (0, "real")


def main() -> None:
    """Load both model tiers, route the input, and print a structured prediction."""
    args = parse_args()
    text = args.text or input("Paste a news headline or article: ").strip()

    # Pick which model tier to use
    use_headline = len(text) < _HEADLINE_THRESHOLD_CHARS
    config_path = args.headline_config if use_headline else args.article_config
    tier = "headline" if use_headline else "article"

    vectorizer, classifier, config, metadata = load_model_artifacts(config_path)
    raw = predict_text(text, vectorizer, classifier, config.get("preprocessing"))

    label, label_name = _apply_threshold(
        raw.fake_probability, config.get("deployment", {})
    )

    print(
        json.dumps(
            {
                "model": metadata["model_name"],
                "model_tier": tier,
                "label": label,
                "label_name": label_name,
                "fake_probability": raw.fake_probability,
                "real_probability": raw.real_probability,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
