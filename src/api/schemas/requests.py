"""Pydantic schemas for API request/response validation."""

from pydantic import BaseModel, Field


class SoilFeaturesRequest(BaseModel):
    """Request schema for soil fertility prediction.

    All nutrient values should be positive numbers from soil analysis.
    """

    N: float = Field(..., ge=0, description="Nitrogen content (kg/ha)")
    P: float = Field(..., ge=0, description="Phosphorus content (kg/ha)")
    K: float = Field(..., ge=0, description="Potassium content (kg/ha)")
    pH: float = Field(..., ge=0, le=14, description="Soil pH value")
    EC: float = Field(..., ge=0, description="Electrical Conductivity (dS/m)")
    OC: float = Field(..., ge=0, description="Organic Carbon (%)")
    S: float = Field(..., ge=0, description="Sulfur content (mg/kg)")
    Zn: float = Field(..., ge=0, description="Zinc content (mg/kg)")
    Fe: float = Field(..., ge=0, description="Iron content (mg/kg)")
    Cu: float = Field(..., ge=0, description="Copper content (mg/kg)")
    Mn: float = Field(..., ge=0, description="Manganese content (mg/kg)")
    B: float = Field(..., ge=0, description="Boron content (mg/kg)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "N": 280,
                    "P": 45,
                    "K": 320,
                    "pH": 6.5,
                    "EC": 0.45,
                    "OC": 0.75,
                    "S": 12,
                    "Zn": 1.2,
                    "Fe": 8.5,
                    "Cu": 1.8,
                    "Mn": 15,
                    "B": 0.5,
                }
            ]
        }
    }

    def to_dict(self) -> dict[str, float]:
        """Convert to feature dictionary for model input."""
        return {
            "N": self.N,
            "P": self.P,
            "K": self.K,
            "pH": self.pH,
            "EC": self.EC,
            "OC": self.OC,
            "S": self.S,
            "Zn": self.Zn,
            "Fe": self.Fe,
            "Cu": self.Cu,
            "Mn": self.Mn,
            "B": self.B,
        }


class SoilTypeResponse(BaseModel):
    """Response schema for soil type classification."""

    class_id: int = Field(..., ge=0, le=3, description="Predicted class index")
    class_name: str = Field(..., description="Human-readable soil type name")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    characteristics: dict[str, str] | None = Field(
        default=None,
        description="Soil type characteristics and recommendations",
    )
    probabilities: dict[str, float] | None = Field(
        default=None,
        description="Probabilities for all classes",
    )


class FertilityResponse(BaseModel):
    """Response schema for soil fertility prediction."""

    class_id: int = Field(..., ge=0, le=2, description="Predicted fertility level")
    class_name: str = Field(..., description="Human-readable fertility level")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    recommendation: str = Field(..., description="Soil management recommendation")
    probabilities: dict[str, float] | None = Field(
        default=None,
        description="Probabilities for all classes",
    )
    warnings: list[str] | None = Field(
        default=None,
        description="Data validation warnings",
    )


class CombinedAnalysisResponse(BaseModel):
    """Response schema for combined soil analysis."""

    soil_type: SoilTypeResponse = Field(..., description="Soil type classification")
    fertility: FertilityResponse = Field(..., description="Fertility prediction")


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    models: dict[str, bool] = Field(
        ...,
        description="Model loading status",
    )


class ErrorResponse(BaseModel):
    """Standard error response schema."""

    detail: str = Field(..., description="Error message")
    error_code: str | None = Field(default=None, description="Error code")
