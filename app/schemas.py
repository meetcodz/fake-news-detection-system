from __future__ import annotations

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
        description="Which model tier was used: 'headline' (< 200 chars) or 'article' (longer).",
    )
