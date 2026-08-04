<h1 align="center">
  <br>
    TruthLens
  <br>
</h1>

<h4 align="center">A production-grade misinformation intelligence platform built with Python, FastAPI, and PyTorch.</h4>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-blue?style=flat-square&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/PyTorch-2.2-EE4C2C?style=flat-square&logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square&logo=huggingface" alt="Transformers">
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License">
</p>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-api">API</a> •
  <a href="#-notebooks">Notebooks</a> •
  <a href="#-project-layout">Project Layout</a>
</p>

---

## Features

| Feature | Description |
|---|---|
| **Multi-Tier Inference** | Automatically routes short headlines to a headline-tuned SVM and longer articles to a full-text GRU/DistilBERT model |
| **Explainable AI (XAI)** | Gradient saliency maps for neural models; coefficient extraction for classical models — every prediction is explainable |
| **Local RAG Evidence** | TF-IDF–powered retrieval-augmented fact-checking against a local curated knowledge base of verified claims |
| **Three Model Families** | Classical (TF-IDF SVM), Deep Learning (BiLSTM/GRU), and Transformer (DistilBERT/DeBERTa) |
| **REST API** | FastAPI backend with automatic OpenAPI docs, CORS, and schema-validated responses |
| **Premium UI** | Newspaper-style React frontend with animated scanning, flagged phrase highlighting, and a live RAG evidence panel |
| **Dockerized** | One-command deployment via `docker compose up` |
| **43 Passing Tests** | Full pytest suite covering data loading, model training, XAI, RAG retrieval, and the REST API |

---

## Architecture

```
Raw Text Input
     │
     ▼
Preprocessing & Dateline Stripping
     │
     ├──► Classical Route (SVM + TF-IDF)
     │         └──► Coefficient-Based XAI
     │
     ├──► Deep Learning Route (GRU/BiLSTM)
     │         └──► Gradient Saliency XAI
     │
     └──► Transformer Route (DistilBERT)
               └──► Gradient Saliency XAI
                         │
                         ▼
                  Uncertain Band Logic (35–65% → "Uncertain")
                         │
                         ▼
                  Local RAG Evidence Retrieval (TF-IDF Cosine Similarity)
                         │
                         ▼
                  REST API Response (label, probability, flagged_phrases, evidence)
                         │
                         ▼
                  TruthLens UI (React + Babel)
```

### Model Stages

| Stage | Model | Accuracy |
|-------|-------|----------|
| Stage 1 | TF-IDF + Logistic Regression (Baseline) | ~94% |
| Stage 2 | Calibrated Linear SVM *(deployed)* | ~96% |
| Stage 3 | Bidirectional GRU *(deployed)* | ~97% |
| Stage 4 | DistilBERT Fine-tuned | ~98%+ |

---

## Quick Start

### Prerequisites

- Python 3.12+
- Git

### 1. Clone & Install

```bash
git clone https://github.com/meetcodz/fake-news-detection-system.git
cd fake-news-detection-system
pip install -e ".[dev,notebook]"
```

### 2. Train Models

```bash
# Stage 1 & 2 — Classical models (fast, CPU-friendly)
python -m src.models.train

# Stage 2 — Classical model comparison
python -m src.models.compare

# Stage 3 — GRU/BiLSTM (requires GPU recommended)
python -m src.models.train_deep

# Stage 4 — DistilBERT fine-tuning
python -m src.models.train_transformer
```

### 3. Launch the API

```bash
uvicorn app.main:app --reload
```

The REST API is now live at `http://127.0.0.1:8000` and the TruthLens UI at `http://127.0.0.1:8000/ui`.

### 4. Docker (One-Command)

```bash
docker compose up --build
```

### 5. Run Tests

```bash
# Use the project virtual environment
.venv-stage2/Scripts/pytest --basetemp=./tmp_pytest
```

---

## API

### `POST /predict`

Classifies a news headline or full article and returns a structured prediction with explainability and fact-check evidence.

**Request**
```json
{
  "text": "Scientists confirm a 15-day global blackout caused by Venus and Jupiter alignment.",
  "model_type": "deep_learning",
  "combine_title_text": true
}
```

**Response**
```json
{
  "label": 1,
  "label_name": "fake",
  "fake_probability": 0.97,
  "real_probability": 0.03,
  "model_tier": "deep_learning",
  "model_type": "deep_learning",
  "model_metadata": { "model_name": "...", "metrics": { "accuracy": 0.97 } },
  "flagged_phrases": ["blackout", "Venus and Jupiter alignment"],
  "evidence": [
    {
      "title": "NASA Confirms 15 Days of Darkness?",
      "verdict": "False",
      "source": "NASA & Snopes",
      "url": "https://www.snopes.com/fact-check/15-days-darkness-november/",
      "similarity_score": 0.89
    }
  ]
}
```

**Model types:** `classical` · `deep_learning` · `transformer`

Interactive docs available at `http://127.0.0.1:8000/docs`

---

## Notebooks

Run from the repo root or any subdirectory — each notebook auto-detects the project root.

| Notebook | Purpose |
|----------|---------|
| `notebooks/01_eda.ipynb` | Dataset exploration and text statistics |
| `notebooks/02_baseline_models.ipynb` | Baseline + classical model comparison |
| `notebooks/03_deep_learning.ipynb` | GRU/BiLSTM training and evaluation |
| `notebooks/04_transformer_finetune.ipynb` | DistilBERT/DeBERTa fine-tuning |

```bash
pip install -e ".[notebook]"
jupyter notebook notebooks/
```

**Dataset:** `data/raw/WELFake_Dataset.csv` — ~72,134 labeled news articles  
**Expected columns:** `text`, `title` (optional), `label` (`0` = real, `1` = fake)

---

## Project Layout

```
├── app/                    FastAPI application (routes, schemas, middleware)
├── src/
│   ├── data/               Dataset loading and preprocessing
│   ├── features/           TF-IDF and feature engineering
│   ├── models/             Training, evaluation, comparison, inference
│   ├── explain/            Explainable AI (gradient saliency, coefficient extraction)
│   └── rag/                Local RAG retriever (TF-IDF cosine similarity)
├── frontend/               TruthLens browser UI (React + Babel, no build step)
├── configs/                YAML configuration for each model stage
├── data/
│   ├── raw/                Raw datasets
│   └── fact_checks.json    Local RAG knowledge base
├── docs/                   Stage-level experiment reports
├── notebooks/              Jupyter experiment notebooks
├── tests/                  pytest test suite (43 tests)
├── utils/                  Shared helpers (logging, config, notebook setup)
├── Dockerfile
└── docker-compose.yml
```

---

## Results

The best classical model selected after Stage 2 evaluation is a **Calibrated Linear SVM** trained on a combined title+body TF-IDF representation.  
Full experiment protocol, metrics, and limitations: [`docs/stage_2_classical_models.md`](docs/stage_2_classical_models.md)

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.12+ |
| ML | scikit-learn, PyTorch, HuggingFace Transformers |
| NLP | NLTK, spaCy |
| Backend | FastAPI, Uvicorn |
| Frontend | React (Babel CDN, no build step) |
| Deployment | Docker, Docker Compose |
| Testing | pytest, pytest-cov |
| Config | YAML + environment variable overrides |

---

