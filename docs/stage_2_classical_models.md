# Stage 2: Classical Model Comparison

## Protocol

This experiment used `data/raw/WELFake_Dataset.csv` with title and body text
combined. After text preprocessing, 72,079 valid documents remained. Exact
cleaned-text duplicates and documents with conflicting duplicate labels were
removed before splitting, leaving 63,625 documents. This leakage guard prevents
the same article from appearing in both train and test data.

The resulting stratified split used 50,900 training and 12,725 test documents
with `random_state=42`. Every model used the same 20,000-feature TF-IDF
representation with unigram/bigram features, `min_df=5`, `max_df=0.9`, and
sublinear term frequency.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Calibrated Linear SVM | **96.98%** | **96.27%** | 97.10% | **96.69%** | **99.56%** |
| XGBoost | 96.77% | 95.45% | **97.52%** | 96.47% | 99.54% |
| Logistic Regression | 95.87% | 94.60% | 96.38% | 95.48% | 99.31% |
| Random Forest | 93.55% | 91.56% | 94.47% | 92.99% | 98.55% |
| Multinomial Naive Bayes | 86.30% | 82.91% | 87.90% | 85.33% | 93.39% |

## Deployment selection

The calibrated Linear SVM is the selected Stage 2 deployment model. It has the
best F1 and ROC-AUC, while `CalibratedClassifierCV` supplies probability output
for credibility scores. Its test confusion matrix is `[[6741, 217], [167,
5600]]` in real/fake label order.

Each model artifact now includes `metadata.json` containing its model settings,
dataset identifier, train/test sample counts, preprocessing/features/training
configuration, UTC training timestamp, and metrics. The selected model can be
loaded through `src.models.inference.load_model_artifacts()`.

## Limitations

Duplicate protection makes this a tougher internal validation than a random
split alone, but it is not external validation. Before claiming real-world
generalization, evaluate the selected model on a separate dataset or a split
grouped by source or publication time.

## Reproduce

```powershell
.\.venv-stage2\Scripts\python.exe -m src.models.compare
```

Generated artifact files remain intentionally ignored by Git; this document and
the code/configuration required to reproduce them are tracked.
