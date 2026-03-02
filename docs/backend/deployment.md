# Deployment Guide

## Development

```bash
# Install all dependencies
pip install -e ".[dev,training]"

# Start dev server with auto-reload
uvicorn src.main:create_app --factory --reload --host 0.0.0.0 --port 8000
```

---

## Docker

### Build and run

```bash
# Build image
docker build -t soil-analysis-api .

# Run container
docker run -p 8000:8000 \
  -v $(pwd)/artifacts:/app/artifacts \
  -e ENVIRONMENT=production \
  soil-analysis-api
```

### Docker Compose (recommended)

```bash
docker compose up --build
```

The `docker-compose.yml` mounts the `artifacts/` directory and
exposes port 8000.

---

## Production Checklist

1. **Model artifacts**: Ensure trained models exist under `artifacts/`.
2. **Environment**: Set `ENVIRONMENT=production` and `DEBUG=false`.
3. **CORS**: Restrict `ALLOWED_ORIGINS` to your frontend domain.
4. **Workers**: Use Gunicorn with Uvicorn workers:

   ```bash
   gunicorn src.main:create_app \
     --factory \
     --workers 4 \
     --worker-class uvicorn.workers.UvicornWorker \
     --bind 0.0.0.0:8000
   ```

5. **Health checks**: Configure your load balancer to poll `/api/v1/health`.
6. **Logging**: Structured JSON logging is enabled by default.
7. **MLflow** (optional): Point `MLFLOW_TRACKING_URI` to your MLflow server
   to enable model registry loading in production.

---

## Testing

```bash
# Run all tests with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test
pytest tests/test_predictions.py -v

# Lint
ruff check src/ tests/
mypy src/
```

---

## CI/CD Notes

1. Run tests and lint on every PR.
2. Train models in a dedicated pipeline; upload artifacts to artifact storage.
3. Build Docker image and push to registry.
4. Deploy with model artifacts mounted or baked into the image.
