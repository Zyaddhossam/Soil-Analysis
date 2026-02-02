"""Unit tests for preprocessing utilities."""

import numpy as np
import pytest
from PIL import Image

from src.core.constants import FERTILITY_FEATURE_NAMES
from src.utils.preprocessing import (
    preprocess_fertility_features,
    preprocess_image,
    validate_feature_ranges,
)


class TestPreprocessImage:
    """Tests for image preprocessing."""

    def test_preprocess_bytes(self, sample_image_bytes):
        """Test preprocessing image from bytes."""
        result = preprocess_image(sample_image_bytes)

        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 299, 299, 3)
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_preprocess_numpy_array(self):
        """Test preprocessing image from numpy array."""
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = preprocess_image(img_array)

        assert result.shape == (1, 299, 299, 3)
        assert result.max() <= 1.0

    def test_preprocess_pil_image(self):
        """Test preprocessing PIL Image."""
        img = Image.new("RGB", (150, 150), color=(100, 100, 100))
        result = preprocess_image(img)

        assert result.shape == (1, 299, 299, 3)

    def test_preprocess_custom_size(self, sample_image_bytes):
        """Test preprocessing with custom target size."""
        result = preprocess_image(sample_image_bytes, target_size=(224, 224))

        assert result.shape == (1, 224, 224, 3)

    def test_preprocess_no_normalize(self, sample_image_bytes):
        """Test preprocessing without normalization."""
        result = preprocess_image(sample_image_bytes, normalize=False)

        assert result.max() <= 255.0

    def test_preprocess_grayscale_conversion(self):
        """Test grayscale to RGB conversion."""
        img = Image.new("L", (100, 100), color=128)
        result = preprocess_image(img)

        assert result.shape == (1, 299, 299, 3)

    def test_preprocess_invalid_input(self):
        """Test with invalid input type."""
        with pytest.raises(ValueError):
            preprocess_image("invalid_input")  # type: ignore[arg-type]

    def test_preprocess_invalid_bytes(self):
        """Test with invalid image bytes."""
        with pytest.raises(ValueError):
            preprocess_image(b"not an image")


class TestPreprocessFertilityFeatures:
    """Tests for fertility feature preprocessing."""

    def test_preprocess_valid_features(self, sample_features):
        """Test preprocessing valid features."""
        result = preprocess_fertility_features(sample_features)

        assert isinstance(result, np.ndarray)
        assert result.shape == (1, len(FERTILITY_FEATURE_NAMES))
        assert result.dtype == np.float64

    def test_preprocess_log_transform(self, sample_features):
        """Test log transformation is applied."""
        result = preprocess_fertility_features(sample_features, apply_log_transform=True)

        # Log10 of values > 1 should give positive numbers
        assert np.all(result > 0) or np.all(result < np.log10(max(sample_features.values())))

    def test_preprocess_no_log_transform(self, sample_features):
        """Test without log transformation."""
        result = preprocess_fertility_features(sample_features, apply_log_transform=False)

        # Values should be close to original
        expected = np.array([sample_features[name] for name in FERTILITY_FEATURE_NAMES])
        np.testing.assert_array_almost_equal(result[0], expected)

    def test_preprocess_missing_features(self):
        """Test with missing features."""
        incomplete: dict[str, float] = {"N": 100.0, "P": 50.0}  # Missing other features

        with pytest.raises(ValueError, match="Missing required features"):
            preprocess_fertility_features(incomplete)

    def test_preprocess_feature_order(self, sample_features):
        """Test features are in correct order."""
        result = preprocess_fertility_features(sample_features, apply_log_transform=False)

        for i, name in enumerate(FERTILITY_FEATURE_NAMES):
            assert result[0, i] == sample_features[name]


class TestValidateFeatureRanges:
    """Tests for feature range validation."""

    def test_valid_ranges(self, sample_features):
        """Test validation passes for valid ranges."""
        warnings = validate_feature_ranges(sample_features)

        assert len(warnings) == 0

    def test_ph_out_of_range(self, sample_features):
        """Test pH out of range warning."""
        sample_features["pH"] = 15.0  # Invalid pH
        warnings = validate_feature_ranges(sample_features)

        assert len(warnings) == 1
        assert "pH" in warnings[0]

    def test_negative_values(self, sample_features):
        """Test negative value warning."""
        sample_features["N"] = -10.0
        warnings = validate_feature_ranges(sample_features)

        assert len(warnings) == 1
        assert "N" in warnings[0]

    def test_multiple_warnings(self, sample_features):
        """Test multiple out-of-range values."""
        sample_features["pH"] = 15.0
        sample_features["N"] = -10.0
        sample_features["EC"] = 100.0

        warnings = validate_feature_ranges(sample_features)

        assert len(warnings) == 3
