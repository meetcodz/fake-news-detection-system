"""Threshold tuning: find optimal fake_probability threshold for headline model."""
from src.models.inference import load_model_artifacts, predict_text

HEADLINES = [
    ("real", "Scientists publish peer-reviewed climate research findings in a new journal."),
    ("real", "The Federal Reserve raised interest rates by 25 basis points today."),
    ("real", "European leaders meet in Brussels to discuss trade policy."),
    ("real", "Scientists working on peer-reviewed climate research findings."),
    ("real", "Prime Minister announces new infrastructure spending bill."),
    ("real", "Stock markets close higher on positive jobs report."),
    ("fake", "BREAKING: Aliens have landed in New York City and are demanding pizza!!"),
    ("fake", "Obama secretly a lizard person, claims anonymous source."),
    ("fake", "Doctors HATE him: local man cures cancer with this one weird trick."),
    ("fake", "Government putting microchips in vaccines to track population."),
    ("fake", "URGENT: Share this before it gets DELETED! The TRUTH about 5G towers!"),
    ("fake", "This shocking video proves mainstream media has been LYING to you."),
]


def evaluate(threshold: float, vec, clf, preproc) -> int:
    correct = 0
    for expected, headline in HEADLINES:
        r = predict_text(headline, vec, clf, preproc)
        predicted = "fake" if r.fake_probability > threshold else "real"
        if predicted == expected:
            correct += 1
    return correct


def main():
    vec, clf, cfg, _ = load_model_artifacts("configs/headline.yaml")
    preproc = cfg.get("preprocessing", {})

    best_threshold, best_score = 0.5, 0
    for t in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        score = evaluate(t, vec, clf, preproc)
        marker = " <-- best" if score > best_score else ""
        print(f"threshold={t:.2f}  correct={score}/{len(HEADLINES)}{marker}")
        if score > best_score:
            best_score, best_threshold = score, t

    print(f"\nBest threshold: {best_threshold}  ({best_score}/{len(HEADLINES)} correct)")
    print()

    # Show predictions at best threshold
    print(f"{'EXPECTED':6s}  {'PREDICTED':6s}  {'FAKE%':6s}  HEADLINE")
    print("-" * 90)
    for expected, headline in HEADLINES:
        r = predict_text(headline, vec, clf, preproc)
        predicted = "fake" if r.fake_probability > best_threshold else "real"
        ok = "OK" if predicted == expected else "WRONG"
        print(f"{expected:6s}  {predicted:6s}  {r.fake_probability * 100:5.1f}%  [{ok}]  {headline[:60]}")


if __name__ == "__main__":
    main()
