"""Integration tests for API endpoints."""

from fastapi import status


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_root_endpoint(self, test_client):
        """Test root endpoint returns API info."""
        response = test_client.get("/")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "docs" in data

    def test_health_endpoint(self, test_client):
        """Test health check endpoint."""
        response = test_client.get("/health")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "models" in data

    def test_ready_endpoint_models_loaded(self, test_client):
        """Test readiness check with loaded models."""
        response = test_client.get("/ready")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "ready"


class TestSoilTypeEndpoint:
    """Tests for soil type classification endpoint."""

    def test_predict_soil_type_success(self, test_client, sample_image_bytes):
        """Test successful soil type prediction."""
        response = test_client.post(
            "/api/v1/predictions/soil-type",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "class_id" in data
        assert "class_name" in data
        assert "confidence" in data

    def test_predict_soil_type_invalid_file_type(self, test_client):
        """Test rejection of invalid file type."""
        response = test_client.post(
            "/api/v1/predictions/soil-type",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_predict_soil_type_with_probabilities(self, test_client, sample_image_bytes):
        """Test prediction with probabilities flag."""
        response = test_client.post(
            "/api/v1/predictions/soil-type",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
            params={"include_probabilities": True},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        # Note: probabilities only included if mock returns them
        assert "class_id" in data

    def test_predict_soil_type_model_not_loaded(self, test_client_no_models, sample_image_bytes):
        """Test error when model not loaded."""
        response = test_client_no_models.post(
            "/api/v1/predictions/soil-type",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE


class TestFertilityEndpoint:
    """Tests for fertility prediction endpoint."""

    def test_predict_fertility_success(self, test_client, sample_features):
        """Test successful fertility prediction."""
        response = test_client.post(
            "/api/v1/predictions/fertility",
            json=sample_features,
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "class_id" in data
        assert "class_name" in data
        assert "confidence" in data
        assert "recommendation" in data

    def test_predict_fertility_missing_features(self, test_client):
        """Test validation error for missing features."""
        incomplete_features = {"N": 100, "P": 50}  # Missing required features

        response = test_client.post(
            "/api/v1/predictions/fertility",
            json=incomplete_features,
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_predict_fertility_invalid_values(self, test_client, sample_features):
        """Test validation error for invalid values."""
        sample_features["pH"] = -5.0  # Invalid negative pH

        response = test_client.post(
            "/api/v1/predictions/fertility",
            json=sample_features,
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_predict_fertility_model_not_loaded(self, test_client_no_models, sample_features):
        """Test error when model not loaded."""
        response = test_client_no_models.post(
            "/api/v1/predictions/fertility",
            json=sample_features,
        )

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE


class TestCombinedAnalysisEndpoint:
    """Tests for combined analysis endpoint."""

    def test_analyze_soil_success(self, test_client, sample_image_bytes, sample_features):
        """Test successful combined analysis."""
        # Build form data with all features
        data = {k: str(v) for k, v in sample_features.items()}

        response = test_client.post(
            "/api/v1/predictions/analyze",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
            data=data,
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "soil_type" in data
        assert "fertility" in data
        assert "class_name" in data["soil_type"]
        assert "class_name" in data["fertility"]
