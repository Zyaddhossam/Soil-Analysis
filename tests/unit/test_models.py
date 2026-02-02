"""Unit tests for model wrappers."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.core.constants import (
    FERTILITY_FEATURE_NAMES,
    FERTILITY_LABELS,
    SOIL_TYPE_LABELS,
)
from src.models.fertility_predictor import FertilityPredictor
from src.models.soil_classifier import SoilClassifier


class TestSoilClassifier:
    """Tests for SoilClassifier."""

    def test_initialization_default(self):
        """Test default initialization."""
        classifier = SoilClassifier()

        assert not classifier.is_loaded
        assert classifier.num_classes == 4
        assert len(classifier.class_names) == 4

    def test_initialization_custom_path(self, tmp_path):
        """Test initialization with custom path."""
        model_path = tmp_path / "model.h5"
        classifier = SoilClassifier(model_path=model_path)

        assert classifier._model_path == model_path

    def test_class_names(self):
        """Test class names property."""
        classifier = SoilClassifier()

        assert classifier.class_names == list(SOIL_TYPE_LABELS.values())

    def test_load_file_not_found(self, tmp_path):
        """Test loading non-existent model file."""
        classifier = SoilClassifier(model_path=tmp_path / "nonexistent.h5")

        with pytest.raises(RuntimeError, match="Model loading failed"):
            classifier.load()

    @patch("tensorflow.keras.models.load_model")
    def test_load_success(self, mock_load_model, tmp_path):
        """Test successful model loading."""
        model_path = tmp_path / "model.h5"
        model_path.touch()

        mock_load_model.return_value = MagicMock()

        classifier = SoilClassifier(model_path=model_path)
        classifier.load()

        assert classifier.is_loaded
        mock_load_model.assert_called_once()

    @patch("tensorflow.keras.models.load_model")
    def test_predict(self, mock_load_model, tmp_path, sample_image_bytes):
        """Test prediction."""
        model_path = tmp_path / "model.h5"
        model_path.touch()

        # Mock model prediction
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.7, 0.1, 0.1, 0.1]])
        mock_load_model.return_value = mock_model

        classifier = SoilClassifier(model_path=model_path)
        classifier.load()

        result = classifier.predict(sample_image_bytes)

        assert "class_id" in result
        assert "class_name" in result
        assert "confidence" in result
        assert result["class_id"] == 0
        assert result["confidence"] == 0.7

    @patch("tensorflow.keras.models.load_model")
    def test_predict_with_probabilities(self, mock_load_model, tmp_path, sample_image_bytes):
        """Test prediction with probabilities."""
        model_path = tmp_path / "model.h5"
        model_path.touch()

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.7, 0.15, 0.1, 0.05]])
        mock_load_model.return_value = mock_model

        classifier = SoilClassifier(model_path=model_path)
        classifier.load()

        result = classifier.predict(sample_image_bytes, return_probabilities=True)

        assert "probabilities" in result
        assert len(result["probabilities"]) == 4


class TestFertilityPredictor:
    """Tests for FertilityPredictor."""

    def test_initialization_default(self):
        """Test default initialization."""
        predictor = FertilityPredictor()

        assert not predictor.is_loaded
        assert predictor.num_classes == 3
        assert predictor.feature_names == FERTILITY_FEATURE_NAMES

    def test_class_names(self):
        """Test class names property."""
        predictor = FertilityPredictor()

        assert predictor.class_names == list(FERTILITY_LABELS.values())

    def test_load_file_not_found(self, tmp_path):
        """Test loading non-existent model file."""
        predictor = FertilityPredictor(model_path=tmp_path / "nonexistent.joblib")

        with pytest.raises(RuntimeError, match="Model loading failed"):
            predictor.load()

    @patch("joblib.load")
    def test_load_success(self, mock_joblib_load, tmp_path):
        """Test successful model loading."""
        model_path = tmp_path / "model.joblib"
        model_path.touch()

        mock_joblib_load.return_value = MagicMock()

        predictor = FertilityPredictor(model_path=model_path)
        predictor.load()

        assert predictor.is_loaded
        mock_joblib_load.assert_called_once()

    @patch("joblib.load")
    def test_predict(self, mock_joblib_load, tmp_path, sample_features):
        """Test prediction."""
        model_path = tmp_path / "model.joblib"
        model_path.touch()

        # Mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1])
        mock_model.predict_proba.return_value = np.array([[0.1, 0.7, 0.2]])
        mock_joblib_load.return_value = mock_model

        predictor = FertilityPredictor(model_path=model_path)
        predictor.load()

        result = predictor.predict(sample_features)

        assert result["class_id"] == 1
        assert result["class_name"] == "Fertile"
        assert result["confidence"] == 0.7

    @patch("joblib.load")
    def test_predict_with_probabilities(self, mock_joblib_load, tmp_path, sample_features):
        """Test prediction with probabilities."""
        model_path = tmp_path / "model.joblib"
        model_path.touch()

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([2])
        mock_model.predict_proba.return_value = np.array([[0.1, 0.2, 0.7]])
        mock_joblib_load.return_value = mock_model

        predictor = FertilityPredictor(model_path=model_path)
        predictor.load()

        result = predictor.predict(sample_features, return_probabilities=True)

        assert "probabilities" in result
        assert len(result["probabilities"]) == 3

    @patch("joblib.load")
    def test_predict_with_warnings(self, mock_joblib_load, tmp_path, sample_features):
        """Test prediction includes warnings for out-of-range values."""
        model_path = tmp_path / "model.joblib"
        model_path.touch()

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1])
        mock_model.predict_proba.return_value = np.array([[0.1, 0.7, 0.2]])
        mock_joblib_load.return_value = mock_model

        predictor = FertilityPredictor(model_path=model_path)
        predictor.load()

        # Set invalid pH
        sample_features["pH"] = 15.0
        result = predictor.predict(sample_features, include_warnings=True)

        assert "warnings" in result
        assert len(result["warnings"]) > 0
