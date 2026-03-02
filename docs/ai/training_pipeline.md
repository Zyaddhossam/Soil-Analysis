# Training Pipeline

## Prerequisites

```bash
# Install training dependencies
pip install -e ".[training]"
```

---

## 1. Data Preparation

### Validate & augment image data

```bash
python -m training.data_collection.prepare_data augment-images \
  --input data/image_dataset/Train \
  --output data/image_dataset_augmented \
  --target-per-class 500
```

### Prepare tabular data (SMOTE, feature engineering, splits)

```bash
python -m training.data_collection.prepare_data all \
  --csv data/dataset1.csv \
  --output data/prepared
```

---

## 2. Soil Type Classifier

### Quick train (EfficientNet-B0, no fine-tuning)

```bash
python -m training.soil_type.train \
  --data data/image_dataset/Train \
  --backbone efficientnet_b0 \
  --epochs 20 \
  --batch-size 32
```

### Full pipeline with fine-tuning + MLflow

```bash
python -m training.soil_type.train \
  --data data/image_dataset/Train \
  --backbone efficientnet_b0 \
  --fine-tune \
  --fine-tune-epochs 10 \
  --fine-tune-lr 1e-5 \
  --epochs 20
```

### Train MobileNetV2 benchmark

```bash
python -m training.soil_type.train \
  --data data/image_dataset/Train \
  --backbone mobilenet_v2 \
  --fine-tune \
  --run-name mobilenet-benchmark
```

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | *required* | Image directory with class subfolders |
| `--output` | `artifacts/soil_classifier` | Output directory |
| `--backbone` | `efficientnet_b0` | `efficientnet_b0`, `mobilenet_v2`, `xception` |
| `--epochs` | 20 | Feature-extraction epochs |
| `--fine-tune` | off | Enable fine-tuning phase |
| `--fine-tune-epochs` | 10 | Fine-tuning epochs |
| `--fine-tune-lr` | 1e-5 | Fine-tuning learning rate |
| `--batch-size` | 32 | Batch size |
| `--dropout` | 0.4 | Dropout rate |
| `--dense-units` | 128 | Dense layer units |
| `--no-mlflow` | off | Disable MLflow tracking |
| `--no-register` | off | Skip model registration |
| `--seed` | 42 | Random seed |

---

## 3. Fertility Predictor

### Quick train (RandomForest, no tuning)

```bash
python -m training.fertility.train \
  --data data/dataset1.csv \
  --algorithm random_forest
```

### With hyperparameter tuning

```bash
python -m training.fertility.train \
  --data data/dataset1.csv \
  --algorithm gradient_boosting \
  --tune
```

### Ensemble training

```bash
python -m training.fertility.train \
  --data data/dataset1.csv \
  --algorithm ensemble \
  --run-name ensemble-v1
```

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | `data/dataset1.csv` | Training CSV path |
| `--output` | `artifacts/fertility_predictor` | Output directory |
| `--algorithm` | `random_forest` | `random_forest`, `gradient_boosting`, `svm`, `knn`, `ensemble` |
| `--tune` | off | Run GridSearchCV hyperparameter tuning |
| `--no-feature-engineering` | off | Skip engineered features |
| `--n-estimators` | 200 | Trees (for RF / GB) |
| `--max-depth` | 12 | Max tree depth |
| `--test-size` | 0.15 | Test split fraction |
| `--no-mlflow` | off | Disable MLflow |
| `--no-register` | off | Skip registration |
| `--seed` | 42 | Random seed |

---

## 4. Outputs

After training, model artifacts are saved to the output directory:

### Soil classifier

```
artifacts/soil_classifier/
├── best_model.h5          # Keras model
├── class_names.json       # Class index mapping
├── model_metadata.json    # Backbone, image size, etc.
├── confusion_matrix.json  # Evaluation data
├── confusion_matrix.png   # Heatmap plot
├── training_history.png   # Accuracy & loss curves
├── per_class_f1.png       # Per-class F1 bar chart
└── logs/                  # TensorBoard logs
```

### Fertility predictor

```
artifacts/fertility_predictor/
├── random_forest_model.joblib  # Trained model
├── scaler.joblib               # StandardScaler (if used)
├── feature_names.json          # Feature ordering
├── confusion_matrix.json       # Evaluation data
├── confusion_matrix.png        # Heatmap plot
├── feature_importances.json    # Feature importances
├── feature_importances.png     # Bar chart
├── classification_report.json  # Per-class metrics
└── roc_curves.png              # ROC curves (OvR)
```
