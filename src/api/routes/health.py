"""Health check endpoints."""

from fastapi import APIRouter, status

from src import __version__
from src.api.dependencies import check_models_loaded
from src.api.schemas.requests import HealthResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health Check",
    description="Check the health status of the API and its models.",
)
async def health_check() -> HealthResponse:
    """Check API health and model status.

    Returns:
        Health status including model loading states.
    """
    models_status = check_models_loaded()

    return HealthResponse(
        status="healthy",
        version=__version__,
        models=models_status,
    )


@router.get(
    "/ready",
    status_code=status.HTTP_200_OK,
    summary="Readiness Check",
    description="Check if the API is ready to serve requests.",
)
async def readiness_check() -> dict:
    """Check if all models are loaded and ready.

    Returns:
        Readiness status.

    Raises:
        HTTPException: If models are not ready.
    """
    models_status = check_models_loaded()
    all_ready = all(models_status.values())

    if not all_ready:
        return {
            "status": "not_ready",
            "message": "Some models are not loaded",
            "models": models_status,
        }

    return {"status": "ready"}
