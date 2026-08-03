from contextlib import asynccontextmanager
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.schemas import PredictionRequest, PredictionResponse, ModelMetadata
from src.data.clean import build_text_column
from src.explain.explain import explain_prediction
from src.models.inference import (
    load_model_artifacts,
    predict_text,
    load_deep_model_artifacts,
    predict_deep_text,
    load_transformer_model_artifacts,
    predict_transformer_text,
)
from src.rag.retriever import LocalRAG

_HEADLINE_THRESHOLD_CHARS = 200

model_state: dict = {}
rag_engine: LocalRAG = LocalRAG()

@asynccontextmanager
async def lifespan(app: FastAPI):
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

        try:
            d_vocab, d_model, d_cfg, d_meta = load_deep_model_artifacts("configs/deep_learning.yaml", "gru")
            model_state["deep_learning"] = {
                "vocabulary": d_vocab,
                "model": d_model,
                "config": d_cfg,
                "metadata": d_meta,
            }
        except FileNotFoundError:
            model_state["deep_learning"] = None

        try:
            t_tok, t_model, t_cfg, t_meta = load_transformer_model_artifacts("configs/transformer.yaml", "distilbert-base-uncased")
            model_state["transformer"] = {
                "tokenizer": t_tok,
                "model": t_model,
                "config": t_cfg,
                "metadata": t_meta,
            }
        except FileNotFoundError:
            model_state["transformer"] = None
    except Exception as exc:
        raise RuntimeError(f"Failed to load model artifacts on startup: {exc}")

    try:
        rag_engine.add_documents_from_json("data/fact_checks.json")
    except Exception as exc:
        print(f"RAG initialization failed (non-fatal): {exc}")

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

app.mount("/ui", StaticFiles(directory="frontend", html=True), name="frontend")

def _resolve_tier(text: str) -> Literal["headline", "article"]:
    return "headline" if len(text.strip()) < _HEADLINE_THRESHOLD_CHARS else "article"

def _apply_threshold(fake_prob: float, deployment_cfg: dict) -> tuple[int, str]:
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
    if not model_state:
        return {"status": "starting", "message": "Model artifacts loading..."}
    dl_state = model_state.get("deep_learning")
    t_state = model_state.get("transformer")
    return {
        "status": "ready",
        "message": "TruthLens API is operational. Deep learning (GRU) is the default.",
        "models": {
            "article": model_state["article"]["metadata"]["model_name"],
            "headline": model_state["headline"]["metadata"]["model_name"],
            "deep_learning": dl_state["metadata"]["model_name"] if dl_state else "not loaded",
            "transformer": t_state["metadata"]["model_name"] if t_state else "not loaded",
        },
        "default_model": "deep_learning (GRU)",
        "routing": f"inputs < {_HEADLINE_THRESHOLD_CHARS} chars → headline model (SVM)",
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
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

    if request.model_type == "deep_learning":
        state = model_state.get("deep_learning")
        if state is None:
            raise HTTPException(
                status_code=503,
                detail="Deep learning model (GRU) is not trained or loaded.",
            )
        tier = "deep_learning"
        try:
            raw = predict_deep_text(
                text=text,
                vocabulary=state["vocabulary"],
                model=state["model"],
                preprocessing_config=state["config"].get("preprocessing"),
                max_sequence_length=state["config"].get("vocabulary", {}).get("max_sequence_length", 300),
            )
        except ValueError as val_err:
            raise HTTPException(status_code=422, detail=str(val_err))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")
    elif request.model_type == "transformer":
        state = model_state.get("transformer")
        if state is None:
            raise HTTPException(
                status_code=503,
                detail="Transformer model (DistilBERT) is not trained or loaded.",
            )
        tier = "transformer"
        try:
            raw = predict_transformer_text(
                text=text,
                tokenizer=state["tokenizer"],
                model=state["model"],
                preprocessing_config=state["config"].get("preprocessing"),
                max_sequence_length=state["config"].get("training", {}).get("max_sequence_length", 256),
            )
        except ValueError as val_err:
            raise HTTPException(status_code=422, detail=str(val_err))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")
    else:
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

    try:
        flagged_phrases = explain_prediction(text, tier, state, label, top_k=5)
    except Exception as exc:
        print(f"Explanation failed: {exc}")
        flagged_phrases = []

    try:
        evidence = rag_engine.retrieve(text, top_k=3)
    except Exception as exc:
        print(f"RAG retrieval failed: {exc}")
        evidence = []

    return PredictionResponse(
        label=label,
        label_name=label_name,
        fake_probability=raw.fake_probability,
        real_probability=raw.real_probability,
        model_metadata=model_meta,
        model_tier=tier,
        model_type=request.model_type,
        flagged_phrases=flagged_phrases,
        evidence=evidence,
    )
