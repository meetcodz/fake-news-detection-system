# Agent Guide — Fake News & Misinformation Detection

This file orients AI agents working in this repository. Full context lives in `PROJECT_CONTEXT.md`.

## What we are building

A **modular misinformation intelligence platform** (portfolio project for Devkriti'26), not a single-classifier notebook. The system should look production-ready and demonstrate clean architecture, ML best practices, explainable AI, API design, and deployment readiness.

### Target capabilities

| Area | Features |
|------|----------|
| Classification | Binary fake vs real; multi-label (propaganda, hate speech, clickbait, AI-generated, satire, misleading context) |
| Scoring | Credibility score, risk level |
| NLP | NER, keyword extraction |
| Intelligence | Real-time verification, explainable predictions |
| Delivery | REST API (FastAPI), dashboard, browser extension |
| Future | RAG fact-checking, knowledge graphs, image verification, GNNs, multilingual |

## Tech stack

- **Language:** Python 3.12+
- **ML:** scikit-learn, PyTorch, HuggingFace Transformers
- **NLP:** spaCy, NLTK, Sentence Transformers
- **API:** FastAPI
- **Viz:** Plotly, Matplotlib
- **DB:** SQLite (dev) → PostgreSQL (prod)
- **Deploy:** Docker
- **Tests:** pytest

## Repository layout

```
data/                  # datasets only (raw/processed gitignored)
src/
  data/                # load + preprocess
  features/            # feature engineering
  models/              # train, infer, evaluate
  verification/        # fact verification
  explainability/      # SHAP, LIME
  nlp/                 # NER, keywords
configs/               # all tunables (paths, hyperparams, API)
notebooks/             # portable Jupyter experiments
utils/                 # shared helpers
tests/                 # pytest
```

**Rule:** No business logic outside `src/`.

## How to work here

### Before coding

1. Explore the repo — it may be early-stage; don't assume modules exist.
2. Read files you will touch or depend on.
3. State a short implementation plan.
4. Ask when requirements are ambiguous.

### While coding

- One responsibility per module; no monolithic scripts.
- Minimal diff — only change what the task needs.
- Type hints, docstrings on public APIs, meaningful errors, centralized logging.
- All config via `configs/` or environment variables — no hardcoded paths or hyperparameters.
- Follow the pipeline: design → implement → review → refactor → test → commit.

### After coding

- Self-review the diff.
- Call out edge cases and missing tests.
- Suggest the next logical step if useful.

## ML progression

Implement and compare models in order:

1. TF-IDF + Logistic Regression (baseline)
2. Classical ML (NB, SVM, RF, XGBoost)
3. Deep learning (FastText, BiLSTM, CNN-LSTM)
4. Transformers (DistilBERT, RoBERTa)

Use the **same evaluation metrics** across models so results are comparable.

## Evaluation checklist

**Binary:** accuracy, precision, recall, F1, ROC AUC, confusion matrix.

**Multi-label:** add Hamming loss, macro F1, per-label precision/recall.

## Notebooks

Build runnable notebooks alongside each stage. They must import from `src/` — never duplicate pipeline logic.

| Notebook | Stage | Status |
|----------|-------|--------|
| `01_eda.ipynb` | Exploration | Ready |
| `02_baseline_models.ipynb` | Baseline + classical | Ready |
| `03_deep_learning.ipynb` | FastText, BiLSTM, CNN-LSTM | Placeholder |
| `04_transformer_finetune.ipynb` | DistilBERT, RoBERTa | Placeholder |

Each notebook bootstraps the project root from `pyproject.toml`. On Colab/Kaggle, set `INSTALL_ON_REMOTE = True` in the first cell.

CLI equivalents:

```bash
python -m src.models.train
python -m src.models.compare
```

## Cursor rules

Persistent rules live in `.cursor/rules/`:

| Rule | Scope |
|------|-------|
| `project-core.mdc` | Always — architecture, folders, workflow |
| `python-standards.mdc` | `**/*.py` — style, logging, config, errors |
| `ml-development.mdc` | `src/data`, `src/features`, `src/models` |
| `testing-standards.mdc` | `tests/**` |

## Git

- One logical change per commit; avoid massive commits.
- Good: `Add preprocessing pipeline`, `Implement TF-IDF features`.
- Do not commit secrets (`.env`), trained weights (`models/`, `*.pkl`), or raw data.

## Long-term bar

Every decision should move the repo toward an **industry-grade ML system**: modular, testable, configurable, explainable, and deployable — not a one-off assignment.
