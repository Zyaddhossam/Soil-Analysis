# Model Architecture

## Overview

The Soil Analysis System uses a **decision-level fusion** approach:
two independent models produce predictions which are combined at the
application layer. No paired image–lab data exists, so joint training
is not feasible.

```
┌──────────────┐         ┌───────────────┐
│  Soil Image  │         │  Lab Values   │
└──────┬───────┘         └───────┬───────┘
       │                         │
       ▼                         ▼
┌───────────────┐         ┌────────────────┐
│  CNN Backbone │         │  ML Classifier │
│ (EfficientNet)│         │  (RF / GB / …) │
└──────┬────────┘         └───────┬────────┘
       │                          │
       ▼                          ▼
  Soil Type                Fertility Level
  (4 classes)              (3 classes)
       │                         │
       └─────────┬───────────────┘
                 ▼
        Decision-Level Fusion
        (Combined Response)
```

---

## 1. Soil Type Classifier (CNN)

### Supported Backbones

| Backbone | Input Size | Params (base) | Use Case |
|----------|-----------|---------------|----------|
| **EfficientNet-B0** | 224 × 224 | 5.3 M | Primary – best accuracy / size trade-off |
| **MobileNetV2** | 224 × 224 | 3.4 M | Benchmark – lightweight, mobile-ready |
| **Xception** | 299 × 299 | 22.9 M | Legacy – original v1 backbone |

### Architecture

```
Input (H × W × 3)
  └─ Backbone (ImageNet pre-trained, frozen initially)
       └─ GlobalAveragePooling2D
            └─ BatchNormalization
                 └─ Dropout (0.4)
                      └─ Dense (128, ReLU)
                           └─ BatchNormalization
                                └─ Dropout (0.4)
                                     └─ Dense (4, Softmax) → Soil Type
```

### Training Strategy

1. **Feature extraction**: Freeze backbone, train head for ~20 epochs
   with Adam (lr = 1e-3) and label smoothing (0.1).
2. **Fine-tuning**: Unfreeze top N layers of backbone, retrain with
   Adam (lr = 1e-5) for ~10 epochs.

### Output Classes

| ID | Label | Description |
|----|-------|-------------|
| 0 | Alluvial Soil | River-deposited, sandy–loamy texture |
| 1 | Black Soil | Clay-rich, high moisture retention |
| 2 | Clay Soil | Fine particles, poor drainage |
| 3 | Red Soil | Iron-oxide rich, well-drained |

---

## 2. Fertility Predictor (Classical ML)

### Supported Algorithms

| Algorithm | Key Strengths |
|-----------|---------------|
| **RandomForest** | Robust, feature importance, handles non-linearity |
| **GradientBoosting** | Higher accuracy potential, sequential learning |
| **SVM** | Good for small datasets, kernel flexibility |
| **KNN** | Simple baseline, distance-based |
| **Ensemble** (VotingClassifier) | Combines RF + GB + SVM via soft voting |

### Features

**12 raw features** from soil lab analysis:

| Feature | Unit | Description |
|---------|------|-------------|
| N | kg/ha | Nitrogen |
| P | kg/ha | Phosphorus |
| K | kg/ha | Potassium |
| pH | – | Soil pH (3.5 – 10) |
| EC | dS/m | Electrical Conductivity |
| OC | % | Organic Carbon |
| S | mg/kg | Sulfur |
| Zn | mg/kg | Zinc |
| Fe | mg/kg | Iron |
| Cu | mg/kg | Copper |
| Mn | mg/kg | Manganese |
| B | mg/kg | Boron |

**5 engineered features** (optional, enabled by default):

| Feature | Formula |
|---------|---------|
| N_P_ratio | N / (P + ε) |
| N_K_ratio | N / (K + ε) |
| NPK_total | N + P + K |
| micro_total | Zn + Fe + Cu + Mn + B |
| OC_pH_interaction | OC × pH |

### Preprocessing

1. Feature engineering (ratios, aggregates, interactions)
2. log₁₀ transformation (all features)
3. Optional StandardScaler (recommended for SVM / KNN)

### Output Classes

| ID | Label |
|----|-------|
| 0 | Less Fertile |
| 1 | Fertile |
| 2 | Highly Fertile |

---

## 3. Decision-Level Fusion

Since the two models operate on independent data sources:

- **Image-only** (Farmer scenario): Soil type is predicted; fertility is
  not predicted and `needs_lab_test = true` is returned.
- **Lab-only** (Engineer scenario): Fertility is predicted; soil type is
  not predicted.
- **Image + Lab** (Combined): Both predictions are returned independently.

No cross-model dependencies exist, ensuring each model can be updated
or replaced without affecting the other.
