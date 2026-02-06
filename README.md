# Soil Analysis 🌱🖼️

A machine learning system for soil type classification from images and soil fertility prediction from nutrient analysis data. Built with MLflow for model versioning and Azure Container Registry for deployment.

## Features

- **Soil Type Classification**: Xception-based CNN for identifying soil types (Alluvial, Black, Clay, Red)
- **Fertility Prediction**: Random Forest model to predict soil fertility levels (Less Fertile, Fertile, Highly Fertile)
- **REST API**: FastAPI-based endpoints for model inference
- **MLflow Integration**: Complete experiment tracking, model registry, and model serving
- **Docker Support**: Containerized deployment with Azure Container Registry
- **Production Ready**: Health checks, logging, error handling, and monitoring

## Quick Start

### Using Docker (Recommended)

```bash
# Pull from Azure Container Registry
az acr login --name soilanalysis
docker pull soilanalysis.azurecr.io/soil-analysis-api:latest

# Run the API
docker run -d -p 8000:8000 soilanalysis.azurecr.io/soil-analysis-api:latest

# Access API documentation
open http://localhost:8000/docs
```

### Local Development

```bash
# Install dependencies
pip install -e ".[all]"

# Start MLflow UI
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000 &

# Train models (they auto-register in MLflow)
python -m training.fertility.train --data data/dataset1.csv
python -m training.soil_type.train --data data/Dataset/Train

# Start API (loads models from MLflow)
python -m uvicorn src.api.main:app --reload --port 8000
```

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
├── data/                        # Training data
├── mlruns/                      # MLflow runs (auto-generated)
├── mlflow.db                    # MLflow tracking database
├── pyproject.toml              # Project configuration
├── Dockerfile                   # Docker image definition
├── docker-compose.yml          # Docker Compose config
└── README.md
```

## Architecture

### MLflow-Based Model Management

This project uses **MLflow** for end-to-end ML lifecycle management:

```
┌─────────────────────────────────────────────────────────────┐
│                     Training Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│  1. Train Model (training/*.py)                             │
│  2. Log Metrics, Params → MLflow                            │
│  3. Save Model → MLflow Model Registry                      │
│  4. No local artifacts needed                               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    MLflow Model Registry                     │
├─────────────────────────────────────────────────────────────┤
│  • fertility-predictor (Random Forest)                      │
│  • soil-classifier (Xception CNN)                           │
│  • Version control & lineage tracking                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Application                     │
├─────────────────────────────────────────────────────────────┤
│  1. Load models from MLflow Registry                        │
│  2. Serve predictions via REST API                          │
│  3. No local model files required                           │
└─────────────────────────────────────────────────────────────┘
```

### Key Benefits

✅ **Centralized Model Management**: All models in MLflow Registry  
✅ **Version Control**: Track all model versions and experiments  
✅ **Reproducibility**: Full lineage of training runs  
✅ **Easy Rollback**: Switch between model versions instantly  
✅ **No Artifact Management**: No manual file copying or storage  
✅ **Scalability**: Deploy to multiple servers with same config  

## Installation

### Prerequisites

- Python 3.10+ (Python 3.11 recommended)
- Docker and Docker Desktop (for containerized deployment)
- Azure CLI (for Azure Container Registry)

### Local Development Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd soil-analysis
```

2. **Create a virtual environment:**
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac
```

3. **Install dependencies:**
```bash
# Install all dependencies including dev and training tools
pip install -e ".[all]"

# Or install only production dependencies
pip install -e .
```

4. **Set up environment (optional):**
```bash
cp .env.example .env
# Edit .env if needed (defaults work for local development)
```

5. **Start MLflow tracking server:**
```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

6. **Train models (or use pre-trained from MLflow):**
```bash
# Train fertility predictor
python -m training.fertility.train --data data/dataset1.csv --experiment "fertility-model"

# Train soil type classifier
python -m training.soil_type.train --data data/Dataset/Train --experiment "soil-type-model"
```

> **Note:** Models are automatically logged to MLflow and registered in the Model Registry. The API loads models directly from MLflow - no manual artifact management needed!

## Usage

### Running the API Locally

**Development mode (with auto-reload):**
```bash
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

**Production mode:**
```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will automatically load models from MLflow Model Registry.

### Running with Docker

**Build the image:**
```bash
docker build -t soil-analysis-api:latest .
```

**Run the container:**
```bash
docker run -d -p 8000:8000 --name soil-api soil-analysis-api:latest
```

**View logs:**
```bash
docker logs -f soil-api
```

**Stop the container:**
```bash
docker stop soil-api
docker rm soil-api
```

### Running from Azure Container Registry

**Pull and run the published image:**
```bash
# Login to Azure Container Registry
az acr login --name soilanalysis

# Pull the image
docker pull soilanalysis.azurecr.io/soil-analysis-api:latest

# Run the container
docker run -d -p 8000:8000 soilanalysis.azurecr.io/soil-analysis-api:latest
```

### Using Docker Compose

```bash
# Start all services (API + MLflow)
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information and health status |
| `/health` | GET | Health check endpoint |
| `/ready` | GET | Readiness check (models loaded) |
| `/api/v1/predict/soil-type` | POST | Classify soil type from image |
| `/api/v1/predict/fertility` | POST | Predict fertility from nutrient data |
| `/api/v1/predict/analyze` | POST | Combined soil analysis |

### API Documentation

Once running, access the interactive documentation:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

### Example Requests

**Soil Type Classification:**
```bash
curl -X POST "http://localhost:8000/api/v1/predict/soil-type" \
  -F "file=@soil_image.jpg"
```

**Response:**
```json
{
  "prediction": {
    "class": "Black Soil",
    "class_id": 1,
    "confidence": 0.95
  },
  "all_probabilities": {
    "Alluvial Soil": 0.02,
    "Black Soil": 0.95,
    "Clay Soil": 0.02,
    "Red Soil": 0.01
  }
}
```

**Fertility Prediction:**
```bash
curl -X POST "http://localhost:8000/api/v1/predict/fertility" \
  -H "Content-Type: application/json" \
  -d '{
    "N": 280, "P": 45, "K": 320, "pH": 6.5,
    "EC": 0.45, "OC": 0.75, "S": 12, "Zn": 1.2,
    "Fe": 8.5, "Cu": 1.8, "Mn": 15, "B": 0.5
  }'
```

**Response:**
```json
{
  "prediction": {
    "fertility_level": "Fertile",
    "level_id": 1,
    "confidence": 0.88
  },
  "recommendations": [
    "Maintain current nutrient levels",
    "Regular soil testing recommended"
  ],
  "nutrient_analysis": {
    "nitrogen": "optimal",
    "phosphorus": "good",
    "potassium": "optimal"
  }
}
```

## Training Models

### Training with MLflow Tracking

All training runs are automatically logged to MLflow with metrics, parameters, and models.

**Fertility Prediction Model:**
```bash
python -m training.fertility.train \
  --data data/dataset1.csv \
  --experiment "fertility-model" \
  --run-name "fertility-v1" \
  --n-estimators 100 \
  --max-depth 10 \
  --test-size 0.2
```

**Soil Type Classification Model:**
```bash
python -m training.soil_type.train \
  --data data/Dataset/Train \
  --experiment "soil-type-model" \
  --run-name "soil-cnn-v1" \
  --epochs 50 \
  --batch-size 32 \
  --learning-rate 0.001
```

### Training Options

**Fertility Model Parameters:**
- `--data`: Path to training CSV file
- `--experiment`: MLflow experiment name
- `--run-name`: Name for this training run
- `--n-estimators`: Number of trees (default: 100)
- `--max-depth`: Maximum tree depth (default: 10)
- `--test-size`: Validation split (default: 0.2)
- `--no-register`: Skip model registration in MLflow

**Soil Type Model Parameters:**
- `--data`: Path to image directory (with class subdirectories)
- `--experiment`: MLflow experiment name
- `--run-name`: Name for this training run
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size (default: 32)
- `--learning-rate`: Learning rate (default: 0.001)
- `--dropout`: Dropout rate (default: 0.4)

### MLflow Tracking UI

Start MLflow UI to view experiments, compare runs, and manage models:

```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Access at: **http://localhost:5000**

### Model Registry

Models are automatically registered in MLflow Model Registry:
- **fertility-predictor**: Random Forest model for fertility prediction
- **soil-classifier**: Xception CNN for soil type classification

The API automatically loads the latest registered version of each model.

## Docker Deployment

### Building the Docker Image

```bash
# Build locally
docker build -t soil-analysis-api:latest .

# Tag for version
docker tag soil-analysis-api:latest soil-analysis-api:v1.0.0
```

### Azure Container Registry

**Push to Azure Container Registry:**

```bash
# Login to Azure
az login
az acr login --name soilanalysis

# Tag for ACR
docker tag soil-analysis-api:latest soilanalysis.azurecr.io/soil-analysis-api:latest
docker tag soil-analysis-api:latest soilanalysis.azurecr.io/soil-analysis-api:v1.0.0

# Push to ACR
docker push soilanalysis.azurecr.io/soil-analysis-api:latest
docker push soilanalysis.azurecr.io/soil-analysis-api:v1.0.0
```

**Pull from Azure Container Registry:**

```bash
# Login
az acr login --name soilanalysis

# Pull the image
docker pull soilanalysis.azurecr.io/soil-analysis-api:latest

# Run
docker run -d -p 8000:8000 \
  --name soil-api \
  soilanalysis.azurecr.io/soil-analysis-api:latest
```

### Image Details

- **Base Image**: Python 3.11-slim
- **Size**: ~4.2GB (includes TensorFlow, scikit-learn, MLflow data)
- **Architecture**: Multi-stage build for optimized size
- **Security**: Runs as non-root user
- **Health Check**: Automatic health monitoring on `/health` endpoint

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

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `APP_NAME` | Application name | `Soil Analysis API` |
| `APP_VERSION` | Application version | `1.0.0` |
| `DEBUG` | Enable debug mode | `false` |
| `ENVIRONMENT` | Environment name | `development` |
| `API_PREFIX` | API route prefix | `/api/v1` |
| `ALLOWED_ORIGINS` | CORS allowed origins | `["http://localhost:3000", "http://localhost:8000"]` |
| `MLFLOW_TRACKING_URI` | MLflow tracking URI | `sqlite:///mlflow.db` |
| `MLFLOW_EXPERIMENT_NAME` | Default experiment name | `soil-analysis` |
| `BATCH_SIZE` | Inference batch size | `32` |
| `CONFIDENCE_THRESHOLD` | Minimum confidence threshold | `0.5` |

### MLflow Configuration

The application uses MLflow for model management:

- **Model Loading**: Models are loaded directly from MLflow Model Registry
- **Tracking**: All training runs are logged to MLflow
- **Registry**: Models are registered with versioning support
- **Database**: SQLite database (`mlflow.db`) stores all experiment data

To use a remote MLflow server, set:
```bash
export MLFLOW_TRACKING_URI=http://your-mlflow-server:5000
```

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