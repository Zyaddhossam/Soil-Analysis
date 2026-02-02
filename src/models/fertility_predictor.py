"""Soil fertility predictor model wrapper.

This module provides a clean interface for the Random Forest classifier
that predicts soil fertility based on nutrient content.
"""

from pathlib import Path

from src.core.config import settings
from src.core.constants import (
    FERTILITY_FEATURE_NAMES,
    FERTILITY_LABELS,
    FERTILITY_RECOMMENDATIONS,
    NUM_FERTILITY_LEVELS,
)
from src.core.logging import get_logger
from src.utils.preprocessing import (
    preprocess_fertility_features,
    validate_feature_ranges,
)

logger = get_logger(__name__)


class FertilityPredictor:
    """Wrapper for the soil fertility prediction model.

    This predictor uses a Random Forest classifier to predict soil
    fertility level based on nutrient content analysis.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        use_mlflow: bool = False,
        mlflow_model_name: str | None = None,
        mlflow_model_version: str | None = None,
    ):
        """Initialize the fertility predictor.

        Args:
            model_path: Path to the saved model (.joblib).
            use_mlflow: Whether to load model from MLflow registry.
            mlflow_model_name: Name of model in MLflow registry.
            mlflow_model_version: Version of model to load.
        """
        self._model = None
        self._model_path: Path | None = (
            Path(model_path) if model_path else settings.fertility_predictor_model_path
        )
        self._use_mlflow = use_mlflow
        self._mlflow_model_name = mlflow_model_name or "fertility-predictor"
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
            logger.info("Fertility predictor loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load fertility predictor: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e

    def _load_from_file(self) -> None:
        """Load model from local file."""
        import joblib

        if self._model_path is None:
            raise RuntimeError("Model path not configured")

        if not self._model_path.exists():
            raise FileNotFoundError(f"Model not found: {self._model_path}")

        logger.info(f"Loading model from: {self._model_path}")
        self._model = joblib.load(self._model_path)

    def _load_from_mlflow(self) -> None:
        """Load model from MLflow registry."""
        from src.utils.mlflow_utils import load_sklearn_model_from_registry

        logger.info(f"Loading model from MLflow: {self._mlflow_model_name}")
        self._model = load_sklearn_model_from_registry(
            model_name=self._mlflow_model_name,
            version=self._mlflow_model_version,
        )

    def predict(
        self,
        features: dict[str, float],
        return_probabilities: bool = False,
        include_warnings: bool = True,
    ) -> dict:
        """Predict soil fertility level.

        Args:
            features: Dictionary of soil nutrient values.
                Required keys: N, P, K, pH, EC, OC, S, Zn, Fe, Cu, Mn, B
            return_probabilities: Whether to include class probabilities.
            include_warnings: Whether to include validation warnings.

        Returns:
            Dictionary containing:
                - class_id: Predicted fertility level (0, 1, or 2)
                - class_name: Human-readable fertility level
                - confidence: Prediction confidence
                - recommendation: Soil management recommendation
                - probabilities: Class probabilities (if requested)
                - warnings: Data validation warnings (if requested)

        Raises:
            ValueError: If required features are missing.
            RuntimeError: If model is not loaded.
        """
        if not self._is_loaded:
            self.load()

        # Validate features
        warnings = validate_feature_ranges(features) if include_warnings else []
        if warnings:
            logger.warning(f"Feature validation warnings: {warnings}")

        # Preprocess features
        processed_features = preprocess_fertility_features(
            features,
            apply_log_transform=True,
        )

        if self._model is None:
            raise RuntimeError("Model not loaded")

        # Run inference
        class_id = int(self._model.predict(processed_features)[0])
        class_name = FERTILITY_LABELS[class_id]

        # Get probabilities if available
        probabilities = None
        confidence = 1.0

        if hasattr(self._model, "predict_proba"):
            proba = self._model.predict_proba(processed_features)[0]
            confidence = float(proba[class_id])
            if return_probabilities:
                probabilities = {
                    FERTILITY_LABELS[i]: round(float(p), 4) for i, p in enumerate(proba)
                }

        result = {
            "class_id": class_id,
            "class_name": class_name,
            "confidence": round(confidence, 4),
            "recommendation": FERTILITY_RECOMMENDATIONS.get(class_id, ""),
        }

        if return_probabilities and probabilities:
            result["probabilities"] = probabilities

        if include_warnings and warnings:
            result["warnings"] = warnings

        logger.debug(f"Prediction: {class_name} ({confidence:.2%})")
        return result

    def predict_batch(
        self,
        feature_list: list[dict[str, float]],
    ) -> list[dict]:
        """Predict fertility for multiple samples.

        Args:
            feature_list: List of feature dictionaries.

        Returns:
            List of prediction dictionaries.
        """
        if not self._is_loaded:
            self.load()

        results = []
        for features in feature_list:
            result = self.predict(
                features,
                return_probabilities=False,
                include_warnings=False,
            )
            results.append(result)

        return results

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded

    @property
    def num_classes(self) -> int:
        """Get number of output classes."""
        return NUM_FERTILITY_LEVELS

    @property
    def class_names(self) -> list[str]:
        """Get list of class names."""
        return list(FERTILITY_LABELS.values())

    @property
    def feature_names(self) -> list[str]:
        """Get list of required feature names."""
        return FERTILITY_FEATURE_NAMES.copy()
