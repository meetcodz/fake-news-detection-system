# PROJECT_CONTEXT.md

# Fake News & Misinformation Detection System

## Purpose

This project is a production-quality Machine Learning and NLP system being built as a portfolio-worthy project for Devkriti'26.

The goal is NOT simply to classify fake news.

The goal is to build a modular misinformation intelligence platform that demonstrates software engineering, machine learning, backend engineering, explainable AI, and production deployment.

Everything should be written as if it were going into production.

---

# Primary Goals

The system should support:

- Binary Fake vs Real classification
- Multi-label misinformation detection
    - Propaganda
    - Hate Speech
    - Clickbait
    - AI Generated Misinformation
    - Satire
    - Misleading Context
- Credibility Score
- Risk Level Prediction
- Explainable Predictions
- Named Entity Extraction
- Keyword Extraction
- Real-time Verification
- Dashboard
- REST API
- Browser Extension

The architecture should remain extensible for future features such as

- RAG Fact Checking
- Knowledge Graph Verification
- Image Verification
- Graph Neural Networks
- Multilingual Support

---

# Development Philosophy

The project should follow professional software engineering practices.

Never generate large monolithic scripts.

Everything must be modular.

Each module should have one responsibility.

Avoid duplicate logic.

Code readability is more important than clever code.

Follow SOLID principles whenever appropriate.

Every important decision should prioritize maintainability over short-term convenience.

---

# Tech Stack

Programming Language

- Python 3.12+

Machine Learning

- Scikit-learn
- PyTorch
- HuggingFace Transformers

NLP

- spaCy
- NLTK
- Sentence Transformers

Backend

- FastAPI

Visualization

- Plotly
- Matplotlib

Database

- PostgreSQL (future)
- SQLite during development if needed

Deployment

- Docker

Testing

- pytest

Version Control

- Git

---

# Project Architecture

The project follows a pipeline architecture.

Raw Input

↓

Data Ingestion

↓

Preprocessing

↓

Feature Engineering

↓

Model Training

↓

Evaluation

↓

Inference

↓

Verification

↓

Explainability

↓

Storage

↓

API

↓

Frontend

Every stage must remain independent.

---

# Folder Responsibilities

data/
Store datasets only.

src/data/
Dataset loading and preprocessing.

src/features/
Feature engineering.

src/models/
Training, inference and evaluation.

src/verification/
Fact verification modules.

src/explainability/
SHAP and LIME.

src/nlp/
NER and keyword extraction.

configs/
Configuration files only.

utils/
Reusable helper functions.

tests/
Unit tests.

No business logic should exist outside src.

---

# Coding Standards

Always

- Use type hints.
- Use descriptive variable names.
- Use docstrings.
- Keep functions focused.
- Prefer composition over inheritance.
- Keep functions under roughly 40 lines where practical.
- Separate configuration from implementation.

Never

- Hardcode paths.
- Hardcode hyperparameters.
- Duplicate code.
- Create circular imports.
- Write unnecessary comments.

Comments should explain WHY, not WHAT.

---

# Error Handling

All public functions should

- Validate inputs.
- Raise meaningful exceptions.
- Log failures.
- Never silently ignore errors.

---

# Logging

Use centralized logging.

Do not use print() for production logic.

All logs should use the logging module.

---

# Configuration

All configurable values must come from configuration files or environment variables.

Examples

- dataset paths
- learning rate
- batch size
- epochs
- API settings
- model selection

Nothing should be hardcoded.

---

# Machine Learning Strategy

Development should happen in stages.

Stage 1

Baseline

- TF-IDF
- Logistic Regression

Stage 2

Classical Models

- Naive Bayes
- SVM
- Random Forest
- XGBoost

Stage 3

Deep Learning

- FastText
- BiLSTM
- CNN-LSTM

Stage 4

Transformers

- DistilBERT
- RoBERTa

Models should always be comparable using identical evaluation metrics.

---

# Evaluation Metrics

Always compute

- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC
- Confusion Matrix

For multi-label models include

- Hamming Loss
- Macro F1
- Per-label Precision
- Per-label Recall

---

# Development Workflow

Every feature follows this order

Design

↓

Implementation

↓

Review

↓

Refactor

↓

Testing

↓

Git Commit

Do not skip steps.

---

# Cursor Behavior Rules

Before writing code

1. Understand the existing repository.
2. Read relevant files.
3. Explain the implementation plan.
4. Ask for clarification if requirements are ambiguous.

When writing code

- Modify only requested files.
- Avoid unnecessary changes.
- Keep commits small.
- Preserve project architecture.
- Explain important design decisions.

After writing code

- Review your own implementation.
- Suggest improvements.
- Mention possible edge cases.
- Recommend tests.

Never rewrite unrelated files.

---

# Documentation

Every module should be understandable without reading the entire repository.

Public functions require docstrings.

Complex algorithms should include implementation notes.

---

# Testing

Every new module should include corresponding pytest tests whenever feasible.

Edge cases should always be tested.

---

# Git Philosophy

One logical change per commit.

Good commit examples

- Add preprocessing pipeline
- Implement TF-IDF features
- Add DistilBERT training
- Create evaluation module

Avoid massive commits.

---

# Long-Term Vision

The final project should resemble an industry Machine Learning system rather than a college assignment.

The repository should demonstrate

- Clean Architecture
- Modular Design
- Production-level Code Quality
- Machine Learning Best Practices
- Explainable AI
- API Development
- Deployment Readiness

Every implementation decision should support these goals.