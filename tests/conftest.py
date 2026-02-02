"""Pytest configuration and fixtures."""

import sys
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Get project root path."""
    return PROJECT_ROOT


@pytest.fixture
def sample_image_bytes() -> bytes:
    """Create sample image bytes for testing."""
    import io

    from PIL import Image

    # Create a simple RGB image
    img = Image.new("RGB", (299, 299), color=(128, 64, 32))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def sample_features() -> dict:
    """Create sample soil features for testing."""
    return {
        "N": 280.0,
        "P": 45.0,
        "K": 320.0,
        "pH": 6.5,
        "EC": 0.45,
        "OC": 0.75,
        "S": 12.0,
        "Zn": 1.2,
        "Fe": 8.5,
        "Cu": 1.8,
        "Mn": 15.0,
        "B": 0.5,
    }


@pytest.fixture
def mock_soil_classifier():
    """Create mock soil classifier."""
    mock = MagicMock()
    mock.is_loaded = True
    mock.predict.return_value = {
        "class_id": 0,
        "class_name": "Alluvial Soil",
        "confidence": 0.95,
        "characteristics": {
            "description": "Test description",
            "suitable_crops": "Test crops",
        },
    }
    return mock


@pytest.fixture
def mock_fertility_predictor():
    """Create mock fertility predictor."""
    mock = MagicMock()
    mock.is_loaded = True
    mock.predict.return_value = {
        "class_id": 1,
        "class_name": "Fertile",
        "confidence": 0.88,
        "recommendation": "Test recommendation",
    }
    return mock


@pytest.fixture
def test_client(
    mock_soil_classifier,
    mock_fertility_predictor,
) -> Generator[TestClient, None, None]:
    """Create FastAPI test client with mocked models."""
    from src.api import dependencies
    from src.api.main import app

    # Patch the dependency functions
    with (
        patch.object(dependencies, "get_soil_classifier", return_value=mock_soil_classifier),
        patch.object(
            dependencies, "get_fertility_predictor", return_value=mock_fertility_predictor
        ),
    ):
        with TestClient(app) as client:
            yield client


@pytest.fixture
def test_client_no_models() -> Generator[TestClient, None, None]:
    """Create FastAPI test client without loaded models."""
    from src.api import dependencies
    from src.api.main import app

    mock_classifier = MagicMock()
    mock_classifier.is_loaded = False

    mock_predictor = MagicMock()
    mock_predictor.is_loaded = False

    with (
        patch.object(dependencies, "get_soil_classifier", return_value=mock_classifier),
        patch.object(dependencies, "get_fertility_predictor", return_value=mock_predictor),
    ):
        with TestClient(app) as client:
            yield client
