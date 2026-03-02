"""Soil type classifier model wrapper.

Supports multiple CNN backbones (EfficientNet-B0, MobileNetV2, Xception)
with per-backbone image sizing and model metadata.
"""

import json
from pathlib import Path

import numpy as np

from src.core.config import settings
from src.core.constants import (
    BACKBONE_IMAGE_SIZES,
    NUM_SOIL_TYPES,
    SOIL_TYPE_CHARACTERISTICS,
    SOIL_TYPE_LABELS,
)
from src.core.logging import get_logger
from src.utils.preprocessing import preprocess_image

logger = get_logger(__name__)


class SoilClassifier:
    """Wrapper for the soil type classification model.

    Automatically resolves the correct input size from model metadata
    or the configured backbone.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        use_mlflow: bool = False,
        mlflow_model_name: str | None = None,
        mlflow_model_version: str | None = None,
    ):
        """Initialize the soil classifier.

        Args:
            model_path: Path to the saved Keras model (.h5).
            use_mlflow: Whether to load model from MLflow registry.
            mlflow_model_name: Name of model in MLflow registry.
            mlflow_model_version: Version of model to load.
        """
        self._model = None
        self._backbone: str = settings.soil_classifier_backbone
        self._image_size: tuple[int, int] = settings.image_size

        if model_path is not None:
            self._model_path: Path = (
                Path(model_path) if not isinstance(model_path, Path) else model_path
            )
        else:
            path_from_settings = settings.soil_classifier_model_path
            if path_from_settings is None:
                raise RuntimeError("Soil classifier model path not configured in settings")
            self._model_path = path_from_settings

        # Try to load metadata for image size
        meta_path = self._model_path.parent / "model_metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                self._backbone = meta.get("backbone", self._backbone)
                size = meta.get("image_size")
                if size and len(size) == 2:
                    self._image_size = tuple(size)  # type: ignore[assignment]
                logger.debug(f"Loaded model metadata: backbone={self._backbone}")
            except Exception:
                pass

        # Fallback: resolve from backbone config
        if self._backbone in BACKBONE_IMAGE_SIZES:
            self._image_size = BACKBONE_IMAGE_SIZES[self._backbone]

        logger.debug(f"SoilClassifier image_size={self._image_size}, backbone={self._backbone}")

        self._use_mlflow = use_mlflow
        self._mlflow_model_name = mlflow_model_name or "soil-classifier"
        self._mlflow_model_version = mlflow_model_version
        self._is_loaded = False

    def load(self) -> None:
        """Load the model into memory.

        Raises:
            FileNotFoundError: If model file doesn't exist.
            RuntimeError: If model loading fails.
        """
        if self._is_loaded:
            logger.debug("Model already loaded, skipping")
            return

        try:
            if self._use_mlflow:
                self._load_from_mlflow()
            else:
                self._load_from_file()

            self._is_loaded = True
            logger.info("Soil classifier loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load soil classifier: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e

    def _load_from_file(self) -> None:
        """Load model from local file."""
        # Import TensorFlow only when needed
        from tensorflow import keras  # type: ignore[import-untyped]

        if not self._model_path.exists():
            raise FileNotFoundError(f"Model not found: {self._model_path}")

        logger.info(f"Loading model from: {self._model_path}")
        self._model = keras.models.load_model(self._model_path)

    def _load_from_mlflow(self) -> None:
        """Load model from MLflow registry."""
        from src.utils.mlflow_utils import load_keras_model_from_registry

        logger.info(f"Loading model from MLflow: {self._mlflow_model_name}")
        self._model = load_keras_model_from_registry(
            model_name=self._mlflow_model_name,
            version=self._mlflow_model_version,
        )

    def predict(
        self,
        image: bytes | np.ndarray,
        return_probabilities: bool = False,
    ) -> dict:
        """Classify a soil image.

        Args:
            image: Image as bytes or numpy array.
            return_probabilities: Whether to include all class probabilities.

        Returns:
            Dictionary containing:
                - class_id: Predicted class index
                - class_name: Human-readable class name
                - confidence: Prediction confidence (0-1)
                - probabilities: Dict of all class probabilities (if requested)
                - characteristics: Soil characteristics for predicted type

        Raises:
            RuntimeError: If model is not loaded.
        """
        if not self._is_loaded:
            self.load()

        # Preprocess image
        processed_image = preprocess_image(
            image,
            target_size=self._image_size,
            normalize=True,
        )

        if self._model is None:
            raise RuntimeError("Model not loaded")

        # Run inference
        predictions = self._model.predict(processed_image, verbose=0)
        probabilities = predictions[0]

        # Get predicted class
        class_id = int(np.argmax(probabilities))
        confidence = float(probabilities[class_id])
        class_name = SOIL_TYPE_LABELS[class_id]

        result = {
            "class_id": class_id,
            "class_name": class_name,
            "confidence": round(confidence, 4),
            "characteristics": SOIL_TYPE_CHARACTERISTICS.get(class_id, {}),
        }

        if return_probabilities:
            result["probabilities"] = {
                SOIL_TYPE_LABELS[i]: round(float(p), 4) for i, p in enumerate(probabilities)
            }

        logger.debug(f"Prediction: {class_name} ({confidence:.2%})")
        return result

    def predict_batch(
        self,
        images: list[bytes | np.ndarray],
    ) -> list[dict]:
        """Classify multiple soil images.

        Args:
            images: List of images as bytes or numpy arrays.

        Returns:
            List of prediction dictionaries.
        """
        if not self._is_loaded:
            self.load()

        # Preprocess all images
        processed_images = np.vstack(
            [preprocess_image(img, target_size=self._image_size) for img in images]
        )

        if self._model is None:
            raise RuntimeError("Model not loaded")

        # Run batch inference
        predictions = self._model.predict(processed_images, verbose=0)

        # Process results
        results = []
        for probs in predictions:
            class_id = int(np.argmax(probs))
            results.append(
                {
                    "class_id": class_id,
                    "class_name": SOIL_TYPE_LABELS[class_id],
                    "confidence": round(float(probs[class_id]), 4),
                    "characteristics": SOIL_TYPE_CHARACTERISTICS.get(class_id, {}),
                }
            )

        return results

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded

    @property
    def num_classes(self) -> int:
        """Get number of output classes."""
        return NUM_SOIL_TYPES

    @property
    def class_names(self) -> list[str]:
        """Get list of class names."""
        return list(SOIL_TYPE_LABELS.values())

    @property
    def backbone(self) -> str:
        """Get the backbone architecture name."""
        return self._backbone

    @property
    def image_size(self) -> tuple[int, int]:
        """Get the expected input image size."""
        return self._image_size
