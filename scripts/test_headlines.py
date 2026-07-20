"""Sanity-check the retrained SVM against real vs fake headlines."""
import json
from src.models.inference import load_model_artifacts, predict_text

HEADLINES = [
    # Should be REAL
    ("real", "Scientists publish peer-reviewed climate research findings in a new journal."),
    ("real", "Scientists working on peer-reviewed climate research findings in a new journal."),
    ("real", "The Federal Reserve raised interest rates by 25 basis points today."),
    ("real", "European leaders meet in Brussels to discuss trade policy."),
    # Should be FAKE
    ("fake", "BREAKING: Aliens have landed in New York City and are demanding pizza!!"),
    ("fake", "Obama secretly a lizard person, claims anonymous source."),
    ("fake", "Doctors HATE him: local man cures cancer with this one weird trick."),
    ("fake", "Government putting microchips in vaccines to track population."),
]


def main():
    vectorizer, classifier, config, _metadata = load_model_artifacts("configs/classical.yaml")
    preprocessing_config = config.get("preprocessing", {})
    correct = 0
    print(f"\n{'EXPECTED':6s}  {'PREDICTED':6s}  {'FAKE%':6s}  HEADLINE")
    print("-" * 90)
    for expected_label, headline in HEADLINES:
        result = predict_text(headline, vectorizer, classifier, preprocessing_config)
        predicted = result.label_name
        fake_pct = result.fake_probability * 100
        ok = "OK" if predicted == expected_label else "WRONG"
        if predicted == expected_label:
            correct += 1
        print(f"{expected_label:6s}  {predicted:6s}  {fake_pct:5.1f}%  [{ok}]  {headline[:65]}")

    print(f"\nResult: {correct}/{len(HEADLINES)} correct")


if __name__ == "__main__":
    main()
