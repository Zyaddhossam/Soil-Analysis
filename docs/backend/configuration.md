# Configuration

The backend is configured via environment variables (or a `.env` file
in the project root). All settings are defined in `src/core/config.py`
using Pydantic Settings.

---

## Environment Variables

### Application

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_NAME` | `Soil Analysis API` | Application name |
| `APP_VERSION` | `1.0.0` | API version |
| `DEBUG` | `false` | Enable debug mode |
| `ENVIRONMENT` | `development` | `development`, `staging`, `production` |

### API

| Variable | Default | Description |
|----------|---------|-------------|
| `API_PREFIX` | `/api/v1` | API route prefix |
| `ALLOWED_ORIGINS` | `["*"]` | CORS allowed origins (JSON array) |

### Model Paths

| Variable | Default | Description |
|----------|---------|-------------|
| `SOIL_CLASSIFIER_MODEL_PATH` | `artifacts/soil_classifier/best_model.h5` | Path to CNN model |
| `FERTILITY_PREDICTOR_MODEL_PATH` | `artifacts/fertility_predictor/random_forest_model.joblib` | Path to ML model |
| `FERTILITY_SCALER_PATH` | auto-detected | Path to feature scaler (if exists) |

### Backbone Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SOIL_CLASSIFIER_BACKBONE` | `efficientnet_b0` | CNN backbone name |
| `USE_FEATURE_ENGINEERING` | `true` | Enable engineered features for fertility |

The image size is resolved automatically from the backbone configuration.
If a `model_metadata.json` file exists alongside the model, it takes precedence.

### MLflow

| Variable | Default | Description |
|----------|---------|-------------|
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | MLflow tracking server |
| `MLFLOW_EXPERIMENT_NAME` | `soil-analysis` | Default experiment |
| `MLFLOW_MODEL_REGISTRY_URI` | (same as tracking URI) | Model registry URI |

### Inference

| Variable | Default | Description |
|----------|---------|-------------|
| `BATCH_SIZE` | `32` | Batch size for inference |
| `CONFIDENCE_THRESHOLD` | `0.5` | Minimum confidence threshold |

---

## .env Example

```env
# App
DEBUG=false
ENVIRONMENT=production

# CORS
ALLOWED_ORIGINS=["https://myapp.example.com"]

# Model paths (absolute or relative to project root)
SOIL_CLASSIFIER_MODEL_PATH=artifacts/soil_classifier/best_model.h5
FERTILITY_PREDICTOR_MODEL_PATH=artifacts/fertility_predictor/random_forest_model.joblib

# Backbone
SOIL_CLASSIFIER_BACKBONE=efficientnet_b0
USE_FEATURE_ENGINEERING=true

# MLflow
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
```

---

## Model Metadata Auto-Detection

When the soil classifier loads, it checks for
`artifacts/soil_classifier/model_metadata.json`:

```json
{
  "backbone": "efficientnet_b0",
  "image_size": [224, 224],
  "num_classes": 4,
  "class_labels": {"0": "Alluvial Soil", "1": "Black Soil", ...}
}
```

This allows the model to self-describe its input requirements,
overriding the configured backbone setting.

Similarly, when a `scaler.joblib` file exists in the fertility
model directory, it is automatically loaded and applied during
inference.
