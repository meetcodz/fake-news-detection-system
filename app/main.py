from contextlib import asynccontextmanager
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import PredictionRequest, PredictionResponse, ModelMetadata
from src.data.clean import build_text_column
from src.models.inference import load_model_artifacts, predict_text

# Global state to hold loaded model artifacts
model_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts on startup to avoid loading latency on incoming requests."""
    try:
        # Default config path is configs/classical.yaml
        # It selects the deployment model configured in the deployment.model_name YAML key
        vectorizer, classifier, config, metadata = load_model_artifacts()
        model_state["vectorizer"] = vectorizer
        model_state["classifier"] = classifier
        model_state["config"] = config
        model_state["metadata"] = metadata
    except Exception as e:
        raise RuntimeError(f"Failed to load model artifacts on startup: {e}")
    yield
    model_state.clear()

app = FastAPI(
    title="TruthLens Misinformation Detection API",
    description="REST API for real-time fake news and misinformation detection using calibrated model predictions.",
    version="0.1.0",
    lifespan=lifespan
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint displaying operational status and active model identifier."""
    if "metadata" not in model_state:
        return {"status": "starting", "message": "Model artifacts loading..."}
    return {
        "status": "ready",
        "message": "TruthLens Misinformation Detection API is operational.",
        "active_model": model_state["metadata"]["model_name"]
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Classify a news document body (and optional title) as real or fake news."""
    if "classifier" not in model_state:
        raise HTTPException(
            status_code=503,
            detail="Model is currently unavailable. Ensure artifacts are trained and loaded."
        )

    text = request.text
    if request.combine_title_text and request.title:
        df = pd.DataFrame([{"title": request.title, "text": request.text}])
        text = build_text_column(
            df,
            text_column="text",
            title_column="title",
            combine_title_text=True
        ).iloc[0]

    try:
        prediction = predict_text(
            text=text,
            vectorizer=model_state["vectorizer"],
            classifier=model_state["classifier"],
            preprocessing_config=model_state["config"].get("preprocessing")
        )
    except ValueError as val_err:
        raise HTTPException(status_code=422, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    metadata = model_state["metadata"]
    model_meta = ModelMetadata(
        model_name=metadata["model_name"],
        trained_at_utc=metadata["trained_at_utc"],
        dataset=metadata["dataset"],
        metrics=metadata["metrics"]
    )

    return PredictionResponse(
        label=prediction.label,
        label_name=prediction.label_name,
        fake_probability=prediction.fake_probability,
        real_probability=prediction.real_probability,
        model_metadata=model_meta
    )
