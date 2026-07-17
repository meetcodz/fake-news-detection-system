"""Classical ML model factories for TF-IDF classification."""

from __future__ import annotations

from typing import Any, Callable

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from utils.logging import get_logger

logger = get_logger(__name__)

ClassifierBuilder = Callable[[dict[str, Any]], Any]

SUPPORTED_CLASSICAL_MODELS = (
    "logistic_regression",
    "naive_bayes",
    "svm",
    "random_forest",
    "xgboost",
)


def build_logistic_regression(config: dict[str, Any]) -> LogisticRegression:
    """Create a logistic regression classifier."""
    return LogisticRegression(
        C=float(config.get("C", 1.0)),
        max_iter=int(config.get("max_iter", 1000)),
        class_weight=config.get("class_weight"),
        random_state=config.get("random_state"),
    )


def build_naive_bayes(config: dict[str, Any]) -> MultinomialNB:
    """Create a multinomial naive Bayes classifier."""
    return MultinomialNB(alpha=float(config.get("alpha", 1.0)))


def build_svm(config: dict[str, Any]) -> Any:
    """Create a linear SVM, optionally calibrated for probability inference."""
    estimator = LinearSVC(
        C=float(config.get("C", 1.0)),
        class_weight=config.get("class_weight"),
        random_state=config.get("random_state"),
        max_iter=int(config.get("max_iter", 2000)),
    )
    if not config.get("calibrate", False):
        return estimator

    return CalibratedClassifierCV(
        estimator=estimator,
        method=str(config.get("calibration_method", "sigmoid")),
        cv=int(config.get("calibration_cv", 5)),
    )


def build_random_forest(config: dict[str, Any]) -> RandomForestClassifier:
    """Create a random forest classifier."""
    return RandomForestClassifier(
        n_estimators=int(config.get("n_estimators", 200)),
        max_depth=config.get("max_depth"),
        class_weight=config.get("class_weight"),
        random_state=config.get("random_state", 42),
        n_jobs=int(config.get("n_jobs", -1)),
    )


def build_xgboost(config: dict[str, Any]) -> Any:
    """Create an XGBoost classifier."""
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for the XGBoost model. Install with: pip install xgboost"
        ) from exc

    return XGBClassifier(
        n_estimators=int(config.get("n_estimators", 200)),
        max_depth=int(config.get("max_depth", 6)),
        learning_rate=float(config.get("learning_rate", 0.1)),
        subsample=float(config.get("subsample", 0.9)),
        colsample_bytree=float(config.get("colsample_bytree", 0.9)),
        eval_metric=config.get("eval_metric", "logloss"),
        random_state=config.get("random_state", 42),
        n_jobs=int(config.get("n_jobs", -1)),
    )


def build_classical_model(model_name: str, config: dict[str, Any]) -> Any:
    """Build a classical classifier by name."""
    builders: dict[str, ClassifierBuilder] = {
        "logistic_regression": build_logistic_regression,
        "naive_bayes": build_naive_bayes,
        "svm": build_svm,
        "random_forest": build_random_forest,
        "xgboost": build_xgboost,
    }

    if model_name not in builders:
        supported = ", ".join(SUPPORTED_CLASSICAL_MODELS)
        raise ValueError(f"Unsupported model '{model_name}'. Supported: {supported}")

    classifier = builders[model_name](config)
    logger.info("Built classical model", extra={"model_name": model_name})
    return classifier
