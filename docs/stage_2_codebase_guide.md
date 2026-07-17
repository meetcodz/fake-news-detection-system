# Stage 2 Codebase Guide

This guide explains the repository as it exists after Stage 2: a configurable,
tested binary news-classification system that trains several classical machine
learning models and deploys a calibrated Linear SVM for local inference.

## What the project does now

Given a CSV of labelled news articles, the project:

1. combines titles and bodies;
2. cleans the text;
3. removes invalid and duplicate documents;
4. converts text to TF-IDF features;
5. trains five classical classifiers on the same split;
6. compares them with standard binary metrics;
7. saves artifacts and metadata; and
8. lets a user classify a pasted article from the terminal.

`0` means **real** and `1` means **fake** throughout the current binary
pipeline.

```mermaid
flowchart LR
    A[WELFake CSV] --> B[loader.py]
    B --> C[clean.py + preprocess.py]
    C --> D[Exact-text deduplication]
    D --> E[Stratified train/test split]
    E --> F[TF-IDF vectorizer]
    F --> G[Five classical models]
    G --> H[evaluate.py]
    H --> I[Artifacts + metadata]
    I --> J[predict.py interactive CLI]
```

## Repository map

| Location | Purpose and Stage 2 status |
| --- | --- |
| `data/raw/` | Local source datasets. Ignored by Git; contains WELFake locally. |
| `data/sample/sample_news.csv` | Small committed CSV used by loader tests. |
| `configs/` | YAML settings: dataset location, preprocessing, features, models, outputs. |
| `src/data/` | Active loading, cleaning, and preprocessing modules. |
| `src/features/` | TF-IDF feature creation. |
| `src/models/` | Active Stage 1/2 training, evaluation, comparison, inference, and CLI modules. |
| `utils/` | Active shared configuration, logging, and notebook helpers. |
| `tests/` | Unit and integration tests. |
| `docs/` | Tracked technical documentation and experiment reports. |
| `models/` | Generated artifacts only; intentionally ignored by Git. |
| `notebooks/` | EDA and model-experiment notebooks. |
| `app/` | Empty FastAPI/Streamlit placeholders; not implemented in Stage 2. |

## Configuration

### `configs/baseline.yaml`

Controls Stage 1: TF-IDF + Logistic Regression. It defines the WELFake path,
preprocessing options, TF-IDF options, Logistic Regression hyperparameters,
the split, and artifact names.

### `configs/classical.yaml`

Controls the complete Stage 2 comparison.

| Section | Meaning |
| --- | --- |
| `dataset` | CSV path, column names, title/body combination, local dataset identifier. |
| `preprocessing` | Lowercasing, URL/HTML removal, and minimum document length. |
| `features` | TF-IDF settings: 20,000 features, 1–2 grams, document-frequency limits. |
| `models` | Models to compare: Logistic Regression, NB, SVM, RF, XGBoost. |
| `model_configs` | Per-model hyperparameters. SVM has five-fold sigmoid calibration enabled. |
| `training` | 20% test size, reproducible seed 42, stratification, exact-text deduplication. |
| `output` | Artifact filenames and `models/classical` output root. |
| `deployment` | Chooses `svm` as the artifact used by inference. |

Environment variables can override any YAML value. For example:

```powershell
$env:FND_DATASET__PATH = 'data/raw/another_dataset.csv'
```

The double underscore means “move into a nested YAML section.” This is handled
by `utils/config.py`.

## Active data modules

### `src/data/clean.py`

- `build_text_column(...)` returns the body text, or `title + '. ' + body` when
  title combination is enabled.
- `drop_invalid_rows(...)` removes missing labels and empty documents, returning
  a reindexed DataFrame and logging the count removed.

### `src/data/loader.py`

This is the active dataset loader.

- `load_dataset(...)` reads CSV data; validates required columns; uses
  `build_text_column`; validates binary labels; and returns parallel
  `list[str]` texts and `list[int]` labels.
- `load_dataset_from_config(...)` resolves the configured dataset path relative
  to the project root and calls `load_dataset`.
- Private validators reject missing columns, null labels, and labels other than
  `0` or `1`.

### `src/data/preprocess.py`

- `preprocess_text(...)` removes URLs and HTML, lowercases text according to
  configuration, collapses whitespace, and rejects documents below
  `min_text_length`.
- `preprocess_dataset(...)` applies this to a corpus while keeping labels
  aligned. Invalid examples are skipped with log warnings.
- `preprocess_corpus(...)` is the no-label convenience wrapper.

### Legacy file: `src/data/load.py`

This older DataFrame-returning loader is not used by the Stage 2 pipeline. New
code should use `src.data.loader`. It should eventually be removed or converted
to a compatibility wrapper after checking whether a notebook depends on it.

## Feature engineering

### `src/features/tfidf.py`

`build_tfidf_vectorizer(config)` creates an unfitted scikit-learn
`TfidfVectorizer`. It only accepts `type: tfidf`, converts the YAML n-gram list
to a tuple, and applies the configured feature limits.

### `src/features/build_features.py`

Empty future placeholder. Stage 2 uses `tfidf.py` directly.

## Model modules

### `src/models/classical.py`

Contains factories for the five Stage 2 estimators:

- `build_logistic_regression`
- `build_naive_bayes`
- `build_svm`
- `build_random_forest`
- `build_xgboost`
- `build_classical_model`

The SVM factory builds `LinearSVC`. When `calibrate: true`, it wraps it in
`CalibratedClassifierCV`. Calibration makes `predict_proba` available, which is
required for the fake/real probability displayed by inference.

### `src/models/pipeline.py`

This is the central reusable training pipeline.

- `resolve_project_path(...)` turns a relative config path into a project path.
- `load_split_data(...)` loads, preprocesses, optionally de-duplicates, and
  splits the dataset.
- `deduplicate_examples(...)` removes repeated cleaned texts. If the same text
  has both labels, it removes that ambiguous example entirely.
- `split_dataset(...)` calls scikit-learn `train_test_split` with stratification.
- `build_tfidf_features(...)` fits TF-IDF on **training text only** and
  transforms test text. This prevents feature leakage.
- `train_tfidf_classifier(...)` fits a supplied estimator, predicts the test
  split, computes metrics, and writes artifacts.
- `_extract_positive_scores(...)` uses probabilities when available, otherwise
  SVM decision scores for ROC-AUC evaluation.
- `_save_artifacts(...)` writes the vectorizer, classifier, `metrics.json`, and
  `metadata.json`.

`metadata.json` records the model class/name, UTC training time, dataset path
and version, sample counts, configurations, and measured metrics.

### `src/models/evaluate.py`

`compute_binary_metrics(...)` returns:

- accuracy;
- precision;
- recall;
- F1;
- a 2×2 confusion matrix; and
- ROC-AUC when scores/probabilities are supplied.

The positive class is fake (`1`). Precision answers “when fake is predicted,
how often is that correct?” Recall answers “how many labelled-fake articles were
found?” F1 balances them. ROC-AUC evaluates ranking quality across thresholds.

### `src/models/compare.py`

`compare_classical_models(...)` is the Stage 2 entry point.

It loads one split and one fitted TF-IDF representation, then trains every
configured model on those exact same features. This makes the comparison fair.
It saves each model in its own directory and writes:

```text
models/classical/model_comparison.csv
models/classical/model_comparison.json
models/classical/<model-name>/classifier.joblib
models/classical/<model-name>/tfidf_vectorizer.joblib
models/classical/<model-name>/metrics.json
models/classical/<model-name>/metadata.json
```

### `src/models/train.py` and `src/models/baseline.py`

`train.py` runs the Stage 1 Logistic Regression baseline through the same shared
pipeline. `baseline.py` re-exports its factory from `classical.py` so there is
only one implementation.

### `src/models/inference.py`

This is model-agnostic inference for classifiers that expose `predict_proba`.

- `load_model_artifacts(...)` loads the deployment model named by
  `configs/classical.yaml` (`svm` by default), plus vectorizer, config, and
  metadata. It rejects an uncalibrated model because confidence scores would be
  misleading.
- `load_baseline_artifacts(...)` preserves the original Stage 1 loading API.
- `predict_text(...)` preprocesses one article, vectorizes it, and returns
  `PredictionResult`.
- `predict_batch(...)` calls the same logic for multiple texts.

### `src/models/predict.py`

The interactive command-line interface. It accepts `--text`, `--config`, and
`--model`; without `--text`, it asks the user to paste one article. Output is
JSON with model name, label, and calibrated fake/real probabilities.

### Compatibility and future placeholders

- `src/models/classic_ml.py` re-exports the classical factories for backward
  compatibility.
- `src/models/deep_learning.py` is empty: Stage 3 placeholder.
- `src/models/transformer.py` is empty: Stage 4 placeholder.

## How the full Stage 2 experiment was made

1. The WELFake CSV was loaded using configured `title`, `text`, and `label`
   columns.
2. 55 too-short documents were excluded after preprocessing.
3. Exact duplicate/ambiguous documents were removed, leaving 63,625 documents.
4. A stratified 80/20 split created 50,900 training and 12,725 test documents.
5. TF-IDF was fit on training text only.
6. The five configured models were trained on the shared sparse feature matrix.
7. The SVM was calibrated with five-fold sigmoid calibration.
8. Every model was evaluated on the untouched test split and persisted.
9. The calibrated SVM won on F1 and ROC-AUC, becoming the configured deployment
   model.

The exact results and limitations are in
[`stage_2_classical_models.md`](stage_2_classical_models.md).

## Running the project

### Environment setup

The verified local environment is `.venv-stage2`:

```powershell
& 'C:\Users\Lenovo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m venv .venv-stage2
.\.venv-stage2\Scripts\python.exe -m pip install -e '.[dev]'
```

The pre-existing `.venv` referenced a removed Python installation in this
workspace, so use `.venv-stage2` unless you create a fresh normal `.venv`.

### Train the Stage 1 baseline

```powershell
.\.venv-stage2\Scripts\python.exe -m src.models.train
```

### Train and compare Stage 2 models

```powershell
.\.venv-stage2\Scripts\python.exe -m src.models.compare
```

This uses the full local WELFake data and can take several minutes. It replaces
the generated artifacts in `models/classical/`.

### Read the comparison

```powershell
Import-Csv models\classical\model_comparison.csv | Format-Table
Get-Content models\classical\svm\metadata.json
```

### Predict one article interactively

```powershell
.\.venv-stage2\Scripts\python.exe -m src.models.predict
```

Paste a headline plus article body on one line. For a text file:

```powershell
.\.venv-stage2\Scripts\python.exe -m src.models.predict --text (Get-Content -Raw .\article.txt)
```

For an explicit model artifact instead of the configured SVM:

```powershell
.\.venv-stage2\Scripts\python.exe -m src.models.predict --model logistic_regression --text "Article text goes here."
```

Input should be a substantive article, not a search query, URL, or short claim.
The model recognises writing patterns in its training data; it is not yet a live
web fact-checking service.

### Run tests

```powershell
.\.venv-stage2\Scripts\python.exe -m pytest tests -q
```

The full suite currently has 23 tests. Some integration tests deliberately use
the full WELFake data, so they take several minutes.

## Test design

| Test file | What it protects |
| --- | --- |
| `conftest.py` | Shared project-root and sample-data fixtures. |
| `test_clean.py` | Title/body combination and invalid-row removal. |
| `test_loader.py` | CSV loading, schema errors, labels, title combination, full dataset availability. |
| `test_preprocess.py` | URL/HTML/lowercase cleaning, configurable flags, short input, label alignment. |
| `test_tfidf.py` | TF-IDF configuration and unsupported feature type errors. |
| `test_evaluate.py` | Metrics and invalid shape rejection. |
| `test_compare.py` | Classical factories, calibrated SVM, deduplication, comparison files, metadata. |
| `test_train_pipeline.py` | Baseline training, artifact/metadata creation, and prediction. |
| `test_notebook.py` | Notebook root discovery. |
| `test_pipeline.py` | Empty placeholder. |

Tests use `tmp_path` for generated artifacts where possible, so they do not
replace the deployable `models/classical` output.

## Supporting and placeholder files

- `utils/config.py`: YAML loader, environment overrides, project-root lookup.
- `utils/logging.py`: central stream logging setup and logger factory.
- `utils/notebook.py`: lets notebooks locate the repository root.
- `pyproject.toml`: package metadata, dependencies, optional extras, and pytest
  settings.
- `requirements.txt`: alternative dependency list for direct pip installs.
- `AGENTS.md` and `PROJECT_CONTEXT.md`: engineering rules, architecture, and
  long-term vision.
- `notebooks/01_eda.ipynb` and `02_baseline_models.ipynb`: Stage 1/2
  experiments that should import project code rather than duplicate it.
- `notebooks/03_deep_learning.ipynb` and `04_transformer_finetune.ipynb`:
  future-stage placeholders.
- `configs/transformer.yaml`, `src/explain/explain.py`, `src/utils/config.py`,
  `src/utils/metrics.py`, `app/*.py`, `Dockerfile`, and `docker-compose.yml`:
  present but empty/future scaffolding, not part of Stage 2 execution.

## Current limitations and next steps

The Stage 2 model is a binary writing-pattern classifier, not a source-aware
fact-checker. Duplicate protection improves internal validation, but true
generalization should next be tested with an external dataset or a split grouped
by publisher/time. The next implementation stages are deep learning, then
transformers, FastAPI, explainability, and verification.
