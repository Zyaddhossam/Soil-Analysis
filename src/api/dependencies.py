"""Dependency injection for FastAPI routes."""

from src.core.logging import get_logger
from src.models.fertility_predictor import FertilityPredictor
from src.models.soil_classifier import SoilClassifier

logger = get_logger(__name__)

# Global model instances (lazy loaded)
_soil_classifier: SoilClassifier | None = None
_fertility_predictor: FertilityPredictor | None = None


def get_soil_classifier() -> SoilClassifier:
    """Get the soil classifier instance.

    Returns:
        Loaded SoilClassifier instance.
    """
    global _soil_classifier

    if _soil_classifier is None:
        logger.info("Initializing SoilClassifier...")
        _soil_classifier = SoilClassifier(
            use_mlflow=False,  # Set to True to use MLflow registry
        )

    # Ensure model is loaded
    if not _soil_classifier.is_loaded:
        try:
            _soil_classifier.load()
        except Exception as e:
            logger.error(f"Failed to load soil classifier: {e}")
            # Return unloaded instance - will fail on predict
            pass

    return _soil_classifier


def get_fertility_predictor() -> FertilityPredictor:
    """Get the fertility predictor instance.

    Returns:
        Loaded FertilityPredictor instance.
    """
    global _fertility_predictor

    if _fertility_predictor is None:
        logger.info("Initializing FertilityPredictor...")
        _fertility_predictor = FertilityPredictor(
            use_mlflow=False,  # Set to True to use MLflow registry
        )

    # Ensure model is loaded
    if not _fertility_predictor.is_loaded:
        try:
            _fertility_predictor.load()
        except Exception as e:
            logger.error(f"Failed to load fertility predictor: {e}")
            # Return unloaded instance - will fail on predict
            pass

    return _fertility_predictor


def check_models_loaded() -> dict[str, bool]:
    """Check which models are currently loaded.

    Returns:
        Dictionary with model names and their loading status.
    """
    return {
        "soil_classifier": _soil_classifier is not None and _soil_classifier.is_loaded,
        "fertility_predictor": (
            _fertility_predictor is not None and _fertility_predictor.is_loaded
        ),
    }


def preload_models() -> None:
    """Preload all models at startup.

    This should be called during application startup to ensure
    models are loaded before receiving requests.
    """
    logger.info("Preloading models...")

    try:
        get_soil_classifier()
        logger.info("Soil classifier loaded")
    except Exception as e:
        logger.warning(f"Failed to preload soil classifier: {e}")

    try:
        get_fertility_predictor()
        logger.info("Fertility predictor loaded")
    except Exception as e:
        logger.warning(f"Failed to preload fertility predictor: {e}")

    logger.info("Model preloading complete")


def cleanup_models() -> None:
    """Cleanup model resources.

    This should be called during application shutdown.
    """
    global _soil_classifier, _fertility_predictor

    logger.info("Cleaning up model resources...")
    _soil_classifier = None
    _fertility_predictor = None
