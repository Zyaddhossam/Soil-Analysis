# Backend API Documentation

This directory contains documentation for the FastAPI backend.

## Contents

| Document | Description |
|----------|-------------|
| [API Reference](api_reference.md) | Endpoints, request/response schemas |
| [Configuration](configuration.md) | Environment variables and settings |
| [Deployment](deployment.md) | Docker, production deployment guide |

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Run development server
uvicorn src.main:create_app --factory --reload --port 8000

# Run tests
pytest tests/ -v --cov=src
```

## Architecture

```
src/
├── __init__.py          # Version
├── main.py              # FastAPI app factory
├── api/
│   ├── dependencies.py  # Dependency injection (model singletons)
│   ├── routes/
│   │   ├── health.py    # /health, /ready endpoints
│   │   └── predictions.py  # /predictions/* endpoints
│   └── schemas/
│       └── requests.py  # Pydantic request/response models
├── core/
│   ├── config.py        # Pydantic Settings (env-based)
│   ├── constants.py     # Enums, labels, feature names
│   └── logging.py       # Structured logging setup
├── models/
│   ├── soil_classifier.py    # CNN inference wrapper
│   └── fertility_predictor.py # ML inference wrapper
└── utils/
    ├── mlflow_utils.py       # MLflow integration
    ├── model_comparison.py   # Model comparison & promotion
    └── preprocessing.py      # Image & feature preprocessing
```
