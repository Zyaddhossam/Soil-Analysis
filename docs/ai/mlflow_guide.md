# MLflow Guide

## Setup

MLflow is configured via environment variables or `.env`:

```env
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
MLFLOW_EXPERIMENT_NAME=soil-analysis
```

Default: SQLite backend in the project root.

### Start the MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open http://localhost:5000 to browse experiments, compare runs, and
inspect artifacts.

---

## Experiment Structure

| Experiment | Description |
|------------|-------------|
| `soil-type-model` | Soil classifier training runs |
| `fertility-model` | Fertility predictor training runs |

Each training run automatically logs:

- **Parameters**: backbone, learning rate, epochs, algorithm, etc.
- **Metrics**: accuracy, F1, ROC-AUC, loss curves (per epoch)
- **Artifacts**: model file, confusion matrix, plots, reports
- **Model**: registered in the MLflow Model Registry

---

## Model Registry

### Registered models

| Model Name | Framework | Description |
|------------|-----------|-------------|
| `soil-classifier` | Keras / TensorFlow | CNN soil type classifier |
| `fertility-predictor` | scikit-learn | Fertility level predictor |

### Loading from registry

```python
from src.utils.mlflow_utils import (
    load_keras_model_from_registry,
    load_sklearn_model_from_registry,
)

# Load latest version
soil_model = load_keras_model_from_registry("soil-classifier")

# Load specific version
fertility_model = load_sklearn_model_from_registry(
    "fertility-predictor", version="3"
)
```

### Promoting models

```python
from src.utils.model_comparison import promote_best_model

# Promote best soil classifier to champion
promote_best_model(
    experiment_name="soil-type-model",
    registered_model_name="soil-classifier",
    metric="f1_macro",
    alias="champion",
)
```

---

## Using MLflow in Training

Both trainers support MLflow out of the box:

```bash
# With MLflow (default)
python -m training.soil_type.train --data data/image_dataset/Train

# Without MLflow
python -m training.soil_type.train --data data/image_dataset/Train --no-mlflow
```

### Programmatic usage

```python
from training.soil_type.train import SoilTypeTrainer
from pathlib import Path

trainer = SoilTypeTrainer(
    data_dir=Path("data/image_dataset/Train"),
    output_dir=Path("artifacts/soil_classifier"),
    backbone="efficientnet_b0",
)

run_id = trainer.run_with_mlflow(
    run_name="efficientnet-v2",
    fine_tune=True,
    epochs=20,
)
print(f"Run ID: {run_id}")
```

---

## Custom Tags

Add tags to organize runs:

```python
import mlflow

with mlflow.start_run():
    mlflow.set_tag("team", "soil-research")
    mlflow.set_tag("data_version", "v2")
    # ... training code
```

---

## Artifact Storage

By default, artifacts are stored alongside the SQLite database.
For production, configure an S3 or GCS artifact store:

```env
MLFLOW_TRACKING_URI=http://mlflow-server:5000
MLFLOW_ARTIFACT_ROOT=s3://my-bucket/mlflow-artifacts
```
