from contextlib import asynccontextmanager
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import PredictionRequest, PredictionResponse, ModelMetadata
from src.data.clean import build_text_column
from src.models.inference import load_model_artifacts, predict_text

# ---------------------------------------------------------------------------
# Global state: two model tiers
#   "headline" — trained on article titles only; accurate for short inputs
#   "article"  — trained on full text + titles; accurate for long documents
# ---------------------------------------------------------------------------
_HEADLINE_THRESHOLD_CHARS = 200  # inputs shorter than this use the headline model

model_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load both model tiers at startup to avoid per-request latency."""
    try:
        a_vec, a_clf, a_cfg, a_meta = load_model_artifacts("configs/classical.yaml")
        h_vec, h_clf, h_cfg, h_meta = load_model_artifacts("configs/headline.yaml")
        model_state["article"] = {
            "vectorizer": a_vec,
            "classifier": a_clf,
            "config": a_cfg,
            "metadata": a_meta,
        }
        model_state["headline"] = {
            "vectorizer": h_vec,
            "classifier": h_clf,
            "config": h_cfg,
            "metadata": h_meta,
        }
    except Exception as exc:
        raise RuntimeError(f"Failed to load model artifacts on startup: {exc}")
    yield
    model_state.clear()


app = FastAPI(
    title="TruthLens Misinformation Detection API",
    description=(
        "REST API for real-time fake news and misinformation detection. "
        "Automatically routes short headlines to a headline-tuned model and "
        "longer articles to a full-text model."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _resolve_tier(text: str) -> Literal["headline", "article"]:
    """Pick the appropriate model tier based on input length."""
    return "headline" if len(text.strip()) < _HEADLINE_THRESHOLD_CHARS else "article"


def _apply_threshold(fake_prob: float, deployment_cfg: dict) -> tuple[int, str]:
    """Apply a configurable threshold and uncertain-band logic to raw probability."""
    threshold = float(deployment_cfg.get("fake_threshold", 0.50))
    band = deployment_cfg.get("uncertain_band", [])
    if band and len(band) == 2:
        lo, hi = float(band[0]), float(band[1])
        if lo < fake_prob < hi:
            return -1, "uncertain"
    if fake_prob >= threshold:
        return 1, "fake"
    return 0, "real"


@app.get("/")
async def root():
    """Health check — shows operational status and loaded model names."""
    if not model_state:
        return {"status": "starting", "message": "Model artifacts loading..."}
    return {
        "status": "ready",
        "message": "TruthLens API is operational.",
        "models": {
            "article": model_state["article"]["metadata"]["model_name"],
            "headline": model_state["headline"]["metadata"]["model_name"],
        },
        "routing": f"inputs < {_HEADLINE_THRESHOLD_CHARS} chars → headline model",
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Classify a news headline or article body as real, fake, or uncertain.

    The API automatically selects the best-suited model tier:
    - **Headline model**: used when input is < 200 characters (title-length)
    - **Article model**: used for longer, full-article inputs

    Predictions in the 35–65% probability band are returned as ``uncertain``
    to avoid overconfident wrong answers on genuinely ambiguous text.
    """
    if not model_state:
        raise HTTPException(
            status_code=503,
            detail="Model is unavailable. Ensure artifacts are trained and loaded.",
        )

    text = request.text
    if request.combine_title_text and request.title:
        df = pd.DataFrame([{"title": request.title, "text": request.text}])
        text = build_text_column(
            df, text_column="text", title_column="title", combine_title_text=True
        ).iloc[0]

    tier = _resolve_tier(text)
    state = model_state[tier]

    try:
        raw = predict_text(
            text=text,
            vectorizer=state["vectorizer"],
            classifier=state["classifier"],
            preprocessing_config=state["config"].get("preprocessing"),
        )
    except ValueError as val_err:
        raise HTTPException(status_code=422, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    label, label_name = _apply_threshold(
        raw.fake_probability, state["config"].get("deployment", {})
    )

    meta = state["metadata"]
    model_meta = ModelMetadata(
        model_name=meta["model_name"],
        trained_at_utc=meta["trained_at_utc"],
        dataset=meta["dataset"],
        metrics=meta["metrics"],
    )

    return PredictionResponse(
        label=label,
        label_name=label_name,
        fake_probability=raw.fake_probability,
        real_probability=raw.real_probability,
        model_metadata=model_meta,
        model_tier=tier,
    )
