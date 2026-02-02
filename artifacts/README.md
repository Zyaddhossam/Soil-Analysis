# Artifacts Directory

This directory contains trained model artifacts.

## Structure

```
artifacts/
├── soil_classifier/
│   └── model.h5           # Xception CNN for soil type classification
└── fertility_predictor/
    └── random_forest_model.joblib  # RF for fertility prediction
```

## Model Sources

Models can be:
1. Trained locally using training scripts
2. Downloaded from MLflow model registry
3. Placed manually from external sources

## Git Ignored

Model files (`.h5`, `.joblib`) are git-ignored due to size.
Use MLflow or DVC for model versioning.
