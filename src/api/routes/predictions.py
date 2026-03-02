"""Prediction endpoints for soil analysis."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status

from src.api.dependencies import get_fertility_predictor, get_soil_classifier
from src.api.schemas.requests import (
    CombinedAnalysisResponse,
    FertilityResponse,
    SoilFeaturesRequest,
    SoilTypeResponse,
    soil_features_from_form,
)
from src.core.logging import get_logger
from src.models.fertility_predictor import FertilityPredictor
from src.models.soil_classifier import SoilClassifier

logger = get_logger(__name__)

router = APIRouter(prefix="/predictions", tags=["Predictions"])


@router.get(
    "/model-info",
    status_code=status.HTTP_200_OK,
    summary="Model Information",
    description="Get information about the loaded models.",
)
async def model_info(
    classifier: SoilClassifier = Depends(get_soil_classifier),
    predictor: FertilityPredictor = Depends(get_fertility_predictor),
) -> dict:
    """Return loaded model metadata.

    Returns:
        Dictionary with model information.
    """
    return {
        "soil_classifier": {
            "backbone": classifier.backbone,
            "image_size": list(classifier.image_size),
            "num_classes": classifier.num_classes,
            "class_names": classifier.class_names,
            "loaded": classifier.is_loaded,
        },
        "fertility_predictor": {
            "num_classes": predictor.num_classes,
            "class_names": predictor.class_names,
            "feature_names": predictor.feature_names,
            "loaded": predictor.is_loaded,
        },
    }


@router.post(
    "/soil-type",
    response_model=SoilTypeResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify Soil Type",
    description="Classify soil type from an uploaded image using CNN model.",
)
async def predict_soil_type(
    file: Annotated[UploadFile, File(description="Soil image (JPEG, PNG)")],
    include_probabilities: Annotated[
        bool,
        Query(description="Include probabilities for all classes"),
    ] = False,
    classifier: SoilClassifier = Depends(get_soil_classifier),
) -> SoilTypeResponse:
    """Classify soil type from image.

    Args:
        file: Uploaded image file.
        include_probabilities: Whether to return all class probabilities.
        classifier: Injected soil classifier instance.

    Returns:
        Soil type classification result.

    Raises:
        HTTPException: If classification fails.
    """
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type: {file.content_type}. Use JPEG or PNG.",
        )

    # Check model is loaded
    if not classifier.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Soil classifier model is not loaded",
        )

    try:
        # Read image data
        image_data = await file.read()
        logger.info(f"Processing image: {file.filename} ({len(image_data)} bytes)")

        # Run prediction
        result = classifier.predict(
            image_data,
            return_probabilities=include_probabilities,
        )

        result["backbone"] = classifier.backbone

        return SoilTypeResponse(**result)

    except ValueError as e:
        logger.error(f"Invalid image data: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed. Please try again.",
        ) from e


@router.post(
    "/fertility",
    response_model=FertilityResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict Soil Fertility",
    description="Predict soil fertility level from nutrient analysis data.",
)
async def predict_fertility(
    features: SoilFeaturesRequest,
    include_probabilities: Annotated[
        bool,
        Query(description="Include probabilities for all classes"),
    ] = False,
    predictor: FertilityPredictor = Depends(get_fertility_predictor),
) -> FertilityResponse:
    """Predict soil fertility from nutrient data.

    Args:
        features: Soil nutrient feature values.
        include_probabilities: Whether to return all class probabilities.
        predictor: Injected fertility predictor instance.

    Returns:
        Fertility prediction result.

    Raises:
        HTTPException: If prediction fails.
    """
    # Check model is loaded
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Fertility predictor model is not loaded",
        )

    try:
        logger.info("Processing fertility prediction request")

        # Run prediction
        result = predictor.predict(
            features.to_dict(),
            return_probabilities=include_probabilities,
            include_warnings=True,
        )

        return FertilityResponse(**result)

    except ValueError as e:
        logger.error(f"Invalid feature data: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed. Please try again.",
        ) from e


@router.post(
    "/analyze",
    response_model=CombinedAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Combined Soil Analysis",
    description="Perform both soil type classification and fertility prediction.",
)
async def analyze_soil(
    file: Annotated[UploadFile, File(description="Soil image (JPEG, PNG)")],
    features: SoilFeaturesRequest = Depends(soil_features_from_form),
    include_probabilities: Annotated[
        bool,
        Query(description="Include probabilities for all classes"),
    ] = False,
    classifier: SoilClassifier = Depends(get_soil_classifier),
    predictor: FertilityPredictor = Depends(get_fertility_predictor),
) -> CombinedAnalysisResponse:
    """Perform combined soil analysis.

    Args:
        file: Uploaded image file for soil type classification.
        features: Soil nutrient values for fertility prediction.
        include_probabilities: Whether to return all class probabilities.
        classifier: Injected soil classifier instance.
        predictor: Injected fertility predictor instance.

    Returns:
        Combined analysis results.

    Raises:
        HTTPException: If analysis fails.
    """
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type: {file.content_type}. Use JPEG or PNG.",
        )

    # Check models are loaded
    if not classifier.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Soil classifier model is not loaded",
        )

    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Fertility predictor model is not loaded",
        )

    try:
        # Read image data
        image_data = await file.read()
        logger.info(f"Processing combined analysis: {file.filename}")

        # Run both predictions
        soil_type_result = classifier.predict(
            image_data,
            return_probabilities=include_probabilities,
        )
        soil_type_result["backbone"] = classifier.backbone

        fertility_result = predictor.predict(
            features.to_dict(),
            return_probabilities=include_probabilities,
            include_warnings=True,
        )

        return CombinedAnalysisResponse(
            soil_type=SoilTypeResponse(**soil_type_result),
            fertility=FertilityResponse(**fertility_result),
        )

    except ValueError as e:
        logger.error(f"Invalid input data: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Analysis failed. Please try again.",
        ) from e
