from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="The news headline or full article body to classify.",
        examples=["Scientists confirm renewable energy breakthrough in new peer-reviewed study."],
    )
    title: str | None = Field(
        None,
        description=(
            "Optional article title. If provided and combine_title_text is True, "
            "it is prepended to the body text before classification."
        ),
    )
    combine_title_text: bool = Field(
        True,
        description="Whether to concatenate title and body text prior to classification.",
    )
    model_type: Literal["classical", "deep_learning", "transformer"] = Field(
        "deep_learning",
        description="Model architecture type to use ('classical' TF-IDF SVM, 'deep_learning' PyTorch GRU, or 'transformer' HuggingFace DistilBERT). Defaults to GRU.",
    )


class ModelMetadata(BaseModel):
    model_name: str = Field(..., description="Name of the deployed model.")
    trained_at_utc: str = Field(..., description="ISO 8601 timestamp of when the model was trained.")
    dataset: dict = Field(..., description="Information about the training dataset.")
    metrics: dict = Field(..., description="Held-out validation metrics (accuracy, precision, recall, F1, ROC-AUC).")


class PredictionResponse(BaseModel):
    label: int = Field(
        ...,
        description="Prediction label: 0 = real, 1 = fake, -1 = uncertain (borderline probability).",
    )
    label_name: str = Field(
        ...,
        description="Human-readable label: 'real', 'fake', or 'uncertain'.",
    )
    fake_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Calibrated probability that the article is fake news.",
    )
    real_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Calibrated probability that the article is real news.",
    )
    model_metadata: ModelMetadata = Field(
        ...,
        description="Metadata of the model tier that produced this prediction.",
    )
    model_tier: str = Field(
        "article",
        description="Which model tier was used: 'headline' (< 200 chars), 'article' (longer), or 'deep_learning'.",
    )
    model_type: str = Field(
        "classical",
        description="Model architecture type: 'classical' or 'deep_learning'.",
    )
    flagged_phrases: list[str] = Field(
        default_factory=list,
        description="Explainable AI output: Top words or phrases contributing to the model's prediction.",
    )
    evidence: list[dict] = Field(
        default_factory=list,
        description="Local RAG output: Factual evidence retrieved from the trusted knowledge base.",
    )

