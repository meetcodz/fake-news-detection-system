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
</p>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-setup-guide">Setup Guide</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-api-endpoints">API Endpoints</a> •
  <a href="#-notebooks">Notebooks</a> •
  <a href="#-project-layout">Project Layout</a> •
  <a href="#-troubleshooting">Troubleshooting</a>
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

## 🛠 Setup Guide

Follow this detailed configuration guide to get your environment running, datasets placed, models trained, and the API deployed.

### 1. Prerequisites
Ensure you have the following installed on your machine:
- **Python 3.12+**
- **Git**
- **Docker & Docker Compose** (Optional: for containerized deployment)

---

### 2. Environment Setup

#### Windows (PowerShell):
```powershell
# Clone the repository
git clone https://github.com/meetcodz/fake-news-detection-system.git
cd fake-news-detection-system

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\Activate.ps1

# Install in editable mode with development, notebook, and api modules
pip install -e ".[dev,notebook,api,deep]"
```

#### Linux / macOS:
```bash
# Clone the repository
git clone https://github.com/meetcodz/fake-news-detection-system.git
cd fake-news-detection-system

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install in editable mode
pip install -e ".[dev,notebook,api,deep]"
```

---

### 3. Dataset Integration
TruthLens utilizes the **WELFake Dataset** containing ~72,134 labeled news articles.

1. Download the dataset from Kaggle: [WELFake Dataset](https://www.kaggle.com/datasets/boltcaster/welfake-dataset).
2. Extract the file and place `WELFake_Dataset.csv` directly in the raw data directory:
   ```
   fake-news-detection-system/data/raw/WELFake_Dataset.csv
   ```
3. *(Optional)* If your dataset is located elsewhere, configure the path via environment variable:
   ```bash
   # Windows (PowerShell)
   $env:FND_DATASET__PATH="path/to/your/WELFake_Dataset.csv"

   # Linux/macOS
   export FND_DATASET__PATH="path/to/your/WELFake_Dataset.csv"
   ```

---

### 4. Running the Training Pipeline
Train the models stage-by-stage using the commands below:

```bash
# Stage 1 (Baseline) & Stage 2 (Classical Models)
python -m src.models.train

# Stage 2 Model Comparison & Metric Evaluation
python -m src.models.compare

# Stage 3 Deep Learning Model (BiLSTM / GRU)
python -m src.models.train_deep

# Stage 4 Transformer Model (DistilBERT)
python -m src.models.train_transformer
```

---

### 5. Running the Application locally
Once the models are trained, launch the FastAPI server:

```bash
uvicorn app.main:app --reload
```
- Interactive API Documentation: [Swagger UI](http://127.0.0.1:8000/docs)
- TruthLens Web UI: [TruthLens Dashboard](http://127.0.0.1:8000/ui)

---

### 6. Running with Docker (One-Command Deployment)
To containerize and run the platform without local setup:

```bash
docker compose up --build
```
This builds the backend container, links resources, maps storage, and exposes the TruthLens UI at `http://localhost:8000/ui`.

---

## Quick Start

For a fast local test run:

```bash
# Install package
pip install -e ".[dev,notebook]"

# Run unit tests (utilizing custom temp dir for Windows compatibility)
python -m pytest tests/ -v --basetemp=./tmp_pytest

# Start Uvicorn Server
uvicorn app.main:app
```

---

## API Endpoints

### `POST /predict`
Submits text or article contents for model prediction and evidence checks.

#### Request Schema:
```json
{
  "text": "Scientists confirm a 15-day global blackout caused by Venus and Jupiter alignment.",
  "model_type": "deep_learning",
  "combine_title_text": true
}
```

#### Response Schema:
```json
{
  "label": 1,
  "label_name": "fake",
  "fake_probability": 0.97,
  "real_probability": 0.03,
  "model_tier": "deep_learning",
  "model_type": "deep_learning",
  "model_metadata": {
    "model_name": "bidirectional_gru",
    "trained_at_utc": "2026-08-01T12:00:00Z",
    "dataset": "WELFake_Dataset.csv",
    "metrics": { "accuracy": 0.971 }
  },
  "flagged_phrases": ["blackout", "Venus and Jupiter alignment"],
  "evidence": [
    {
      "title": "NASA Confirms 15 Days of Darkness?",
      "verdict": "False",
      "source": "NASA & Snopes",
      "url": "https://www.snopes.com/fact-check/15-days-darkness-november/",
      "similarity_score": 0.892
    }
  ]
}
```

---

## Notebooks

Run from the repo root or any subdirectory — each notebook auto-detects the project root.

| Notebook | Purpose |
|----------|---------|
| `notebooks/01_eda.ipynb` | Dataset exploration and text statistics |
| `notebooks/02_baseline_models.ipynb` | Baseline + classical model comparison |
| `notebooks/03_deep_learning.ipynb` | GRU/BiLSTM training and evaluation |
| `notebooks/04_transformer_finetune.ipynb` | DistilBERT/DeBERTa fine-tuning |

To run:
```bash
pip install -e ".[notebook]"
jupyter notebook notebooks/
```

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

## 🛠 Troubleshooting

#### Pytest PermissionError on Windows
Running pytest on Windows might throw a `PermissionError` when writing to the default system temp directory. Resolve this by specifying a local test temp directory:
```bash
python -m pytest tests/ -v --basetemp=./tmp_pytest
```

#### Out of Memory (OOM) Errors on GPU
When training Stage 4 Transformer model on custom hardware with limited VRAM:
1. Open `configs/transformer.yaml`.
2. Lower the `batch_size` (e.g. to `8` or `4`).
3. Decrease the `max_sequence_length` configuration parameter.

#### Missing Model Artifacts on Startup
Ensure that you have run the training pipeline before starting the FastAPI app. If a specific model family is not trained (e.g. Stage 4 transformers), the API will fall back gracefully or report the model as unavailable rather than crashing.

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

## 📄 License
This project is licensed under the MIT License.
