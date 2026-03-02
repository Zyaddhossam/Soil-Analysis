# Evaluation & Metrics

## Metrics Computed

### Common (both models)

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correct predictions / total |
| Precision (macro) | Average precision across classes |
| Recall (macro) | Average recall across classes |
| F1 (macro) | Harmonic mean of precision & recall |
| F1 (weighted) | F1 weighted by class support |
| ROC-AUC (OvR, macro) | One-vs-Rest AUC averaged |
| Confusion Matrix | True vs predicted per class |
| Classification Report | Per-class precision, recall, F1, support |

### Soil classifier specific

| Metric | Description |
|--------|-------------|
| Training history | Accuracy & loss curves per epoch |
| Per-class F1 | Bar chart of F1 per soil type |

### Fertility predictor specific

| Metric | Description |
|--------|-------------|
| Feature importances | Tree-based feature importance scores |
| 5-fold CV accuracy | Cross-validated accuracy on training set |
| ROC curves | Per-class ROC with AUC values |

---

## Evaluation Artifacts

All artifacts are saved to the model output directory and logged to MLflow.

### JSON artifacts

- `confusion_matrix.json` – matrix values + class labels
- `classification_report.json` – per-class precision / recall / F1
- `feature_importances.json` – feature name → importance (fertility only)

### Plot artifacts

- `confusion_matrix.png` – Seaborn heatmap
- `training_history.png` – Accuracy & loss vs epoch (soil type only)
- `per_class_f1.png` – Bar chart (soil type only)
- `feature_importances.png` – Horizontal bar chart (fertility only)
- `roc_curves.png` – Per-class ROC curves (fertility only)

---

## Model Comparison

Use the comparison utility to rank experiments:

```python
from src.utils.model_comparison import compare_runs, generate_comparison_report
from pathlib import Path

# Compare soil type runs
runs = compare_runs("soil-type-model", metric="f1_macro", top_n=5)
for r in runs:
    print(f"{r['run_name']}: f1={r['metrics'].get('f1_macro', 'N/A')}")

# Generate JSON report
generate_comparison_report(
    "soil-type-model",
    Path("reports/soil_type_comparison.json"),
    metric="f1_macro",
)
```

### Promote best model

```python
from src.utils.model_comparison import promote_best_model

version = promote_best_model(
    "soil-type-model",
    "soil-classifier",
    metric="f1_macro",
)
print(f"Promoted version: {version}")
```

---

## Interpreting Results

### Confusion matrix

A good model should have strong values along the diagonal.
Off-diagonal values indicate misclassification patterns.
For example, Clay ↔ Black confusion is common due to visual similarity.

### Feature importances (fertility)

High-importance features (typically pH, OC, N) drive predictions.
Low-importance features may indicate redundancy or noisy data.
Engineered features (e.g., `NPK_total`, `OC_pH_interaction`) often
rank highly, validating the domain knowledge baked into them.

### ROC-AUC

AUC ≥ 0.95 indicates excellent discrimination.
AUC < 0.80 suggests the model struggles to separate that class.
