# AI / Machine Learning Documentation

This directory contains comprehensive documentation for the AI/ML components
of the Soil Analysis System.

## Contents

| Document | Description |
|----------|-------------|
| [Architecture](architecture.md) | Model architectures, backbones, and design decisions |
| [Training Pipeline](training_pipeline.md) | How to train, tune, and evaluate models |
| [Evaluation](evaluation.md) | Metrics, artifacts, and model comparison |
| [MLflow Guide](mlflow_guide.md) | Experiment tracking, model registry, and promotion |

## Quick Start

### Train the soil type classifier (EfficientNet-B0)

```bash
python -m training.soil_type.train \
  --data data/image_dataset/Train \
  --backbone efficientnet_b0 \
  --fine-tune \
  --epochs 20
```

### Train the fertility predictor (enhanced)

```bash
python -m training.fertility.train \
  --data data/dataset1.csv \
  --algorithm random_forest \
  --tune
```

### Compare model runs

```python
from src.utils.model_comparison import compare_runs
runs = compare_runs("soil-type-model", metric="f1_macro", top_n=5)
```
