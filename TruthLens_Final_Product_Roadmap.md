# TruthLens -- Final Product Feature Roadmap

## Project Vision

**TruthLens** is an AI-powered Fake News Detection & Evidence
Verification System that not only predicts whether a news article is
fake or real, but also explains the prediction and retrieves supporting
evidence from trusted sources.

------------------------------------------------------------------------

# Core Features

## 1. News Input

-   Paste article text
-   Upload PDF, TXT, or DOCX
-   Paste article URL (optional)
-   Separate headline and body input

------------------------------------------------------------------------

## 2. Preprocessing

-   HTML removal
-   URL removal
-   Whitespace normalization
-   Configurable lowercasing
-   Tokenization
-   Duplicate detection

------------------------------------------------------------------------

## 3. Fake News Prediction

### Classical Models

-   Logistic Regression
-   Naive Bayes
-   Random Forest
-   XGBoost
-   Linear SVM

### Deep Learning

-   BiLSTM
-   GRU

### Transformer

-   DeBERTa (default production model)
-   Optional: BERT for comparison

Users can switch between models for evaluation.

------------------------------------------------------------------------

## 4. Confidence Estimation

Display: - Prediction - Confidence score - Reliability level

Example:

-   Prediction: **Likely Fake**
-   Confidence: **96.7%**
-   Reliability: **High**

------------------------------------------------------------------------

## 5. Multi-Model Voting

Display predictions from every model.

Example:

-   Logistic Regression → Fake
-   Naive Bayes → Fake
-   SVM → Fake
-   BiLSTM → Fake
-   DeBERTa → Fake

Then show: - Consensus (e.g. 5/5 models agree) - Warning when models
disagree

------------------------------------------------------------------------

## 6. Explainable AI

Implement: - SHAP - LIME - Attention visualization (Transformer)

Highlight: - Important words - Important phrases - Sections contributing
to the prediction

------------------------------------------------------------------------

## 7. Local RAG (Evidence Retrieval)

Use: - Sentence Transformers - FAISS - Trusted document collection

Pipeline:

Article → Embedding → FAISS Search → Retrieve Top-k Trusted Articles

No commercial AI APIs required.

------------------------------------------------------------------------

## 8. Evidence Comparison

For every prediction, show supporting evidence from retrieved documents.

Example:

-   Reuters: No supporting evidence
-   WHO: Contradicts official guidance
-   NASA: No matching announcement

------------------------------------------------------------------------

## 9. Source Credibility Analysis

If a URL is available, analyze:

-   Domain
-   HTTPS
-   Credibility score
-   Known misinformation status
-   Domain age (optional)

------------------------------------------------------------------------

## 10. Writing Style Analysis

Estimate:

-   Clickbait level
-   Emotional tone
-   Fear language
-   Exaggeration
-   Political bias indicators

------------------------------------------------------------------------

## 11. Similar Trusted Articles

Display the most similar retrieved articles with similarity scores.

------------------------------------------------------------------------

## 12. Robustness Stress Testing

Automatically test prediction stability against:

-   Typos
-   Synonym replacement
-   Lowercase conversion
-   Sentence reordering

Display prediction changes.

------------------------------------------------------------------------

## 13. Error Analysis Dashboard

Include:

-   Confusion Matrix
-   ROC Curve
-   Precision-Recall Curve
-   False Positives
-   False Negatives

------------------------------------------------------------------------

## 14. Model Comparison Dashboard

Compare every model on:

-   Accuracy
-   Precision
-   Recall
-   F1-score
-   ROC-AUC
-   Training time
-   Inference time
-   Model size

------------------------------------------------------------------------

## 15. Training Dashboard

Show:

-   Training loss
-   Validation loss
-   Accuracy curves
-   Learning curves

------------------------------------------------------------------------

## 16. PDF Report Generation

Generate a downloadable report containing:

-   Prediction
-   Confidence
-   Retrieved evidence
-   Similar articles
-   Explainability results
-   Model used
-   Final verdict

------------------------------------------------------------------------

## 17. Analysis History

Maintain previous analyses.

Features:

-   Search history
-   Re-open reports
-   Compare previous predictions

------------------------------------------------------------------------

## 18. Admin Panel (Optional)

-   Import trusted documents
-   Rebuild FAISS index
-   Retrain models
-   View logs

------------------------------------------------------------------------

## 19. Professional UI

Pages:

-   Home
-   Analyze
-   Model Comparison
-   Evidence
-   Dashboard
-   History
-   About

------------------------------------------------------------------------

# Final Architecture

User Input → Preprocessing → Fake News Classifier → Confidence
Estimation → Explainability → Local RAG (FAISS + Sentence Transformers)
→ Evidence Comparison → Report Generation → Dashboard

------------------------------------------------------------------------

# Priority Order

## Phase 1

-   Classical ML
-   Comparison dashboard

## Phase 2

-   BiLSTM
-   GRU

## Phase 3

-   DeBERTa

## Phase 4

-   Explainability (SHAP/LIME)

## Phase 5

-   Local RAG (FAISS + Sentence Transformers)

## Phase 6

-   PDF reports
-   Dashboard polish
-   Robustness testing

------------------------------------------------------------------------

# Three Biggest "Wow" Features

1.  **Transformer-based classifier (DeBERTa)**
2.  **Local RAG using Sentence Transformers + FAISS with trusted
    evidence retrieval**
3.  **Explainable AI using SHAP/LIME/Attention visualization**

These three features together transform the project from a simple fake
news classifier into a professional AI-powered misinformation analysis
platform.
