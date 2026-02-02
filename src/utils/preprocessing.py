"""Image and data preprocessing utilities."""

import io

import numpy as np
from PIL import Image

from src.core.constants import DEFAULT_IMAGE_SIZE, FERTILITY_FEATURE_NAMES
from src.core.logging import get_logger

logger = get_logger(__name__)


def preprocess_image(
    image_data: bytes | np.ndarray | Image.Image,
    target_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    normalize: bool = True,
) -> np.ndarray:
    """Preprocess an image for model inference.

    Args:
        image_data: Image as bytes, numpy array, or PIL Image.
        target_size: Target size (height, width).
        normalize: Whether to normalize pixel values to [0, 1].

    Returns:
        Preprocessed image array with shape (1, height, width, channels).

    Raises:
        ValueError: If image format is invalid.
    """
    # Convert to PIL Image
    if isinstance(image_data, bytes):
        try:
            image = Image.open(io.BytesIO(image_data))
        except Exception as e:
            logger.error(f"Failed to decode image bytes: {e}")
            raise ValueError("Invalid image data") from e
    elif isinstance(image_data, np.ndarray):
        image = Image.fromarray(image_data)
    elif isinstance(image_data, Image.Image):
        image = image_data
    else:
        raise ValueError(f"Unsupported image type: {type(image_data)}")

    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize
    image = image.resize(target_size, Image.Resampling.LANCZOS)

    # Convert to numpy array
    img_array = np.array(image, dtype=np.float32)

    # Normalize
    if normalize:
        img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    logger.debug(f"Preprocessed image shape: {img_array.shape}")
    return img_array


def preprocess_fertility_features(
    features: dict[str, float],
    apply_log_transform: bool = True,
) -> np.ndarray:
    """Preprocess soil nutrient features for fertility prediction.

    Args:
        features: Dictionary of feature name to value.
        apply_log_transform: Whether to apply log10 transformation.

    Returns:
        Feature array with shape (1, num_features).

    Raises:
        ValueError: If required features are missing.
    """
    # Validate required features
    missing_features = set(FERTILITY_FEATURE_NAMES) - set(features.keys())
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")

    # Extract features in correct order
    feature_values = [features[name] for name in FERTILITY_FEATURE_NAMES]
    feature_array = np.array(feature_values, dtype=np.float64).reshape(1, -1)

    # Apply log transformation (matching training preprocessing)
    if apply_log_transform:
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        feature_array = np.log10(feature_array + epsilon)

    logger.debug(f"Preprocessed features shape: {feature_array.shape}")
    return feature_array


def validate_feature_ranges(features: dict[str, float]) -> list[str]:
    """Validate that feature values are within expected ranges.

    Args:
        features: Dictionary of feature name to value.

    Returns:
        List of warning messages for out-of-range values.
    """
    # Expected ranges based on typical soil analysis values
    expected_ranges = {
        "N": (0, 500),  # Nitrogen (kg/ha)
        "P": (0, 200),  # Phosphorus (kg/ha)
        "K": (0, 1000),  # Potassium (kg/ha)
        "pH": (3.5, 10.0),  # pH scale
        "EC": (0, 10),  # Electrical Conductivity (dS/m)
        "OC": (0, 10),  # Organic Carbon (%)
        "S": (0, 100),  # Sulfur (mg/kg)
        "Zn": (0, 50),  # Zinc (mg/kg)
        "Fe": (0, 500),  # Iron (mg/kg)
        "Cu": (0, 50),  # Copper (mg/kg)
        "Mn": (0, 200),  # Manganese (mg/kg)
        "B": (0, 10),  # Boron (mg/kg)
    }

    warnings = []
    for name, value in features.items():
        if name in expected_ranges:
            min_val, max_val = expected_ranges[name]
            if value < min_val or value > max_val:
                warnings.append(f"{name}={value} is outside expected range [{min_val}, {max_val}]")

    return warnings
