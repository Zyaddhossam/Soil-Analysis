# Soil Analysis 🌱🖼️

A machine learning system for soil type classification from images and soil fertility prediction from nutrient analysis data.

## Features

- **Soil Type Classification**: CNN-based image classification to identify soil types (Alluvial, Black, Clay, Red)
- **Fertility Prediction**: Random Forest model to predict soil fertility levels (Less Fertile, Fertile, Highly Fertile)
- **REST API**: FastAPI-based endpoints for model inference
- **MLflow Integration**: Experiment tracking and model registry
- **Docker Support**: Containerized deployment

## Project Structure
```
soil-analysis/
├── src/                          # Source code
│   ├── api/                      # FastAPI application
│   │   ├── main.py              # Application factory
│   │   ├── dependencies.py      # Dependency injection
│   │   ├── routes/              # API endpoints
│   │   │   ├── health.py        # Health check endpoints
│   │   │   └── predictions.py   # Prediction endpoints
│   │   └── schemas/             # Pydantic models
│   │       └── requests.py      # Request/Response schemas
│   ├── core/                    # Core configuration
│   │   ├── config.py            # Settings management
│   │   ├── constants.py         # Class mappings & constants
│   │   └── logging.py           # Logging configuration
│   ├── models/                  # Model wrappers
│   │   ├── soil_classifier.py   # CNN model wrapper
│   │   └── fertility_predictor.py # RF model wrapper
│   └── utils/                   # Utilities
│       ├── preprocessing.py     # Data preprocessing
│       └── mlflow_utils.py      # MLflow helpers
├── training/                    # Training scripts
│   ├── soil_type/
│   │   └── train.py            # Soil type model training
│   └── fertility/
│       └── train.py            # Fertility model training
├── tests/                       # Test suite
│   ├── unit/                    # Unit tests
│   └── integration/             # Integration tests
├── artifacts/                   # Model artifacts (git-ignored)
├── data/                        # Data files
├── pyproject.toml              # Project configuration
├── Dockerfile                   # Docker image
├── docker-compose.yml          # Docker Compose config
└── README.md
```

## Installation

### Prerequisites

- Python 3.10+
- pip or uv package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd soil-analysis
```

2. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -e ".[dev]"
```

4. Copy environment configuration:
```bash
cp .env.example .env
```

5. Place model files in the `artifacts/` directory:
   - `artifacts/soil_classifier/model.h5`
   - `artifacts/fertility_predictor/random_forest_model.joblib`

## Usage

### Running the API

**Development mode:**
```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

**Production mode:**
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

**With Docker:**
```bash
docker-compose up -d
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/ready` | GET | Readiness check |
| `/api/predictions/soil-type` | POST | Classify soil from image |
| `/api/predictions/fertility` | POST | Predict fertility from nutrients |
| `/api/predictions/analyze` | POST | Combined analysis |

### API Documentation

Once running, access the interactive documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Example Requests

**Soil Type Classification:**
```bash
curl -X POST "http://localhost:8000/api/predictions/soil-type" \
  -F "file=@soil_image.jpg"
```

**Fertility Prediction:**
```bash
curl -X POST "http://localhost:8000/api/predictions/fertility" \
  -H "Content-Type: application/json" \
  -d '{
    "N": 280, "P": 45, "K": 320, "pH": 6.5,
    "EC": 0.45, "OC": 0.75, "S": 12, "Zn": 1.2,
    "Fe": 8.5, "Cu": 1.8, "Mn": 15, "B": 0.5
  }'
```

## Training Models

### Fertility Model

```bash
python -m training.fertility.train `
  --data data/dataset1.csv `
  --output artifacts/fertility_predictor `
  --n-estimators 100 `
  --max-depth 10
```

### Soil Type Model

```bash
python -m training.soil_type.train `
  --data path/to/soil_images `
  --output artifacts/soil_classifier `
  --epochs 10 `
  --batch-size 32 `
  --fine-tune
```

### MLflow Tracking

View experiment tracking:
```bash
mlflow ui --host 0.0.0.0 --port 5000
```

Access at: http://localhost:5000

## Testing

Run all tests:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=src --cov-report=html
```

Run specific test files:
```bash
pytest tests/unit/test_preprocessing.py
pytest tests/integration/test_api.py
```

## Configuration

Configuration is managed through environment variables or `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `DEBUG` | Enable debug mode | `false` |
| `ENVIRONMENT` | Environment name | `development` |
| `API_PREFIX` | API route prefix | `/api` |
| `MLFLOW_TRACKING_URI` | MLflow tracking URI | `sqlite:///mlflow.db` |
| `SOIL_CLASSIFIER_MODEL_PATH` | Path to soil classifier | `artifacts/soil_classifier/model.h5` |
| `FERTILITY_PREDICTOR_MODEL_PATH` | Path to fertility model | `artifacts/fertility_predictor/random_forest_model.joblib` |

## Class Definitions

### Soil Types
| Class | ID | Description |
|-------|-----|-------------|
| Alluvial Soil | 0 | Formed by river deposits |
| Black Soil | 1 | Rich in clay, retains moisture |
| Clay Soil | 2 | Fine particles, holds nutrients |
| Red Soil | 3 | Rich in iron oxides |

### Fertility Levels
| Class | ID | Description |
|-------|-----|-------------|
| Less Fertile | 0 | Requires significant improvement |
| Fertile | 1 | Good fertility, maintain with regular care |
| Highly Fertile | 2 | Excellent, well-suited for most crops |

## Development

### Code Quality

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

## License

MIT License - see LICENSE file for details.