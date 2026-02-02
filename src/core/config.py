"""Application configuration using Pydantic settings.

This module provides centralized configuration management using environment
variables with sensible defaults for development.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict  # type: ignore[import-untyped]

# Compute project root at module level
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "Soil Analysis API"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, description="Enable debug mode")
    environment: Literal["development", "staging", "production"] = "development"

    # API Configuration
    api_prefix: str = "/api/v1"
    allowed_origins: list[str] = Field(
        default=["*"],
        description="CORS allowed origins",
    )

    # Model Paths
    base_dir: Path = _PROJECT_ROOT
    soil_classifier_model_path: Path | None = Field(
        default=None,
        description="Path to soil type classifier model (model.h5)",
    )
    fertility_predictor_model_path: Path | None = Field(
        default=None,
        description="Path to fertility predictor model (.joblib)",
    )

    @field_validator("soil_classifier_model_path", "fertility_predictor_model_path", mode="before")
    @classmethod
    def empty_string_to_none(cls, v: Any) -> Any:
        """Convert empty strings to None for model paths."""
        if v == "" or v is None:
            return None
        return v

    # MLflow Configuration
    mlflow_tracking_uri: str = Field(
        default="sqlite:///mlflow.db",
        description="MLflow tracking server URI",
    )
    mlflow_experiment_name: str = Field(
        default="soil-analysis",
        description="Default MLflow experiment name",
    )
    mlflow_model_registry_uri: str | None = Field(
        default=None,
        description="MLflow model registry URI (defaults to tracking URI)",
    )

    # Image Processing
    image_size: tuple[int, int] = (299, 299)
    image_channels: int = 3

    # Inference
    batch_size: int = Field(default=32, ge=1, le=256)
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for predictions",
    )

    def model_post_init(self, __context: Any) -> None:
        """Set default paths and MLflow registry URI if not provided."""
        # Set default model paths
        if self.soil_classifier_model_path is None:
            self.soil_classifier_model_path = _PROJECT_ROOT / "artifacts" / "soil_classifier" / "best_model.h5"
        
        if self.fertility_predictor_model_path is None:
            self.fertility_predictor_model_path = _PROJECT_ROOT / "artifacts" / "fertility_predictor" / "random_forest_model.joblib"
        
        if self.mlflow_model_registry_uri is None:
            self.mlflow_model_registry_uri = self.mlflow_tracking_uri


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Returns:
        Settings: Application settings singleton.
    """
    return Settings()


# Convenience export
settings = get_settings()
