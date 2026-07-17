# Fake News & Misinformation Detection

Modular ML/NLP platform for fake news and misinformation detection. See `PROJECT_CONTEXT.md` and `AGENTS.md` for full architecture and conventions.

## Quick start

```bash
pip install -e ".[dev,notebook]"

# Stage 1 baseline
python -m src.models.train

# Stage 2 classical model comparison
python -m src.models.compare

# Tests
python -m pytest tests/ -v
```

## Notebooks (portable)

Run from the repo root or any subdirectory — each notebook auto-detects the project root.

| Notebook | Purpose |
|----------|---------|
| `notebooks/01_eda.ipynb` | Dataset exploration and text statistics |
| `notebooks/02_baseline_models.ipynb` | Baseline + classical model comparison |
| `notebooks/03_deep_learning.ipynb` | Stage 3 deep learning (upcoming) |
| `notebooks/04_transformer_finetune.ipynb` | Stage 4 transformers (upcoming) |

### Local Jupyter

```bash
pip install -e ".[notebook]"
jupyter notebook notebooks/
```

### Colab / Kaggle / remote

1. Upload or clone this repository.
2. Open a notebook.
3. Run the first setup cell — it installs the package editable and imports from `src/`.

Point `configs/*.yaml` at your dataset path, or override with:

```bash
set FND_DATASET__PATH=data/raw/your_dataset.csv
```

Expected CSV columns: `text`, `label` (`0` = real, `1` = fake). For WELFake, configs also use `title` combined with `text`.

**Current dataset:** `data/raw/WELFake_Dataset.csv` (~72k articles)

## Project layout

```
src/data/         Load and preprocess datasets
src/features/     TF-IDF and future feature builders
src/models/       Training, evaluation, comparison, inference
configs/          YAML configuration
notebooks/        Runnable experiments
tests/            pytest suite
```
