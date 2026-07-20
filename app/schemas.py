from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="The full body text of the article to classify.",
        examples=["Scientists confirm renewable energy breakthrough in new peer-reviewed study."]
    )
    title: str | None = Field(
        None,
        description="Optional title of the article. If provided and combine_title_text is true, it will be prepended to the body text."
    )
    combine_title_text: bool = Field(
        True,
        description="Whether to combine the title and body text prior to running classification."
    )

class ModelMetadata(BaseModel):
    model_name: str = Field(..., description="The name of the deployed model.")
    trained_at_utc: str = Field(..., description="ISO timestamp indicating when the model was trained.")
    dataset: dict = Field(..., description="Information about the dataset used for training.")
    metrics: dict = Field(..., description="Validation metrics (Accuracy, Precision, Recall, F1, ROC-AUC).")

class PredictionResponse(BaseModel):
    label: int = Field(..., description="Binary prediction label: 0 for Real, 1 for Fake.")
    label_name: str = Field(..., description="Human-readable prediction label ('real' or 'fake').")
    fake_probability: float = Field(..., description="Calibrated prediction probability of the article being fake.")
    real_probability: float = Field(..., description="Calibrated prediction probability of the article being real.")
    model_metadata: ModelMetadata = Field(..., description="Metadata of the serving model version.")
