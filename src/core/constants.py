"""Application constants and class mappings.

This module centralizes all constant values and class definitions
to ensure consistency across the application.
"""

from enum import IntEnum
from typing import Final


class SoilType(IntEnum):
    """Soil type classification classes.

    Based on visual characteristics from soil images.
    """

    ALLUVIAL = 0
    BLACK = 1
    CLAY = 2
    RED = 3

    @classmethod
    def get_name(cls, value: int) -> str:
        """Get human-readable name for a class value."""
        return SOIL_TYPE_LABELS.get(value, "Unknown")

    @classmethod
    def from_name(cls, name: str) -> "SoilType":
        """Get enum from human-readable name."""
        name_lower = name.lower().replace(" ", "_").replace("soil", "").strip("_")
        mapping = {
            "alluvial": cls.ALLUVIAL,
            "black": cls.BLACK,
            "clay": cls.CLAY,
            "red": cls.RED,
        }
        return mapping.get(name_lower, cls.ALLUVIAL)


class FertilityLevel(IntEnum):
    """Soil fertility classification classes.

    Based on nutrient content analysis (N, P, K, pH, etc.).
    """

    LESS_FERTILE = 0
    FERTILE = 1
    HIGHLY_FERTILE = 2

    @classmethod
    def get_name(cls, value: int) -> str:
        """Get human-readable name for a class value."""
        return FERTILITY_LABELS.get(value, "Unknown")

    @classmethod
    def from_name(cls, name: str) -> "FertilityLevel":
        """Get enum from human-readable name."""
        name_lower = name.lower().replace(" ", "_")
        mapping = {
            "less_fertile": cls.LESS_FERTILE,
            "fertile": cls.FERTILE,
            "highly_fertile": cls.HIGHLY_FERTILE,
        }
        return mapping.get(name_lower, cls.FERTILE)


# Human-readable labels for soil types
SOIL_TYPE_LABELS: Final[dict[int, str]] = {
    SoilType.ALLUVIAL: "Alluvial Soil",
    SoilType.BLACK: "Black Soil",
    SoilType.CLAY: "Clay Soil",
    SoilType.RED: "Red Soil",
}

# Human-readable labels for fertility levels
FERTILITY_LABELS: Final[dict[int, str]] = {
    FertilityLevel.LESS_FERTILE: "Less Fertile",
    FertilityLevel.FERTILE: "Fertile",
    FertilityLevel.HIGHLY_FERTILE: "Highly Fertile",
}

# Reverse mappings (name to class index)
SOIL_TYPE_NAME_TO_INDEX: Final[dict[str, int]] = {
    name: idx for idx, name in SOIL_TYPE_LABELS.items()
}

FERTILITY_NAME_TO_INDEX: Final[dict[str, int]] = {
    name: idx for idx, name in FERTILITY_LABELS.items()
}

# Number of classes
NUM_SOIL_TYPES: Final[int] = len(SoilType)
NUM_FERTILITY_LEVELS: Final[int] = len(FertilityLevel)

# Feature names for fertility prediction
FERTILITY_FEATURE_NAMES: Final[list[str]] = [
    "N",  # Nitrogen
    "P",  # Phosphorus
    "K",  # Potassium
    "pH",  # Soil pH
    "EC",  # Electrical Conductivity
    "OC",  # Organic Carbon
    "S",  # Sulfur
    "Zn",  # Zinc
    "Fe",  # Iron
    "Cu",  # Copper
    "Mn",  # Manganese
    "B",  # Boron
]

NUM_FERTILITY_FEATURES: Final[int] = len(FERTILITY_FEATURE_NAMES)

# Image preprocessing constants
DEFAULT_IMAGE_SIZE: Final[tuple[int, int]] = (224, 224)
IMAGE_CHANNELS: Final[int] = 3

# Per-backbone image sizes
BACKBONE_IMAGE_SIZES: Final[dict[str, tuple[int, int]]] = {
    "efficientnet_b0": (224, 224),
    "mobilenet_v2": (224, 224),
    "xception": (299, 299),
}

# Engineered feature names (added during training v2)
ENGINEERED_FEATURE_NAMES: Final[list[str]] = [
    "N_P_ratio",
    "N_K_ratio",
    "NPK_total",
    "micro_total",
    "OC_pH_interaction",
]

# Recommendations based on fertility level
FERTILITY_RECOMMENDATIONS: Final[dict[int, str]] = {
    FertilityLevel.LESS_FERTILE: (
        "This soil requires significant improvement. Consider adding organic matter, "
        "compost, and balanced fertilizers. Soil testing is recommended to identify "
        "specific nutrient deficiencies."
    ),
    FertilityLevel.FERTILE: (
        "This soil has good fertility. Maintain with regular organic matter additions "
        "and balanced fertilization based on crop requirements."
    ),
    FertilityLevel.HIGHLY_FERTILE: (
        "Excellent soil fertility! This soil is well-suited for most crops. "
        "Focus on maintaining organic matter levels and avoid over-fertilization."
    ),
}

# Soil type characteristics
SOIL_TYPE_CHARACTERISTICS: Final[dict[int, dict[str, str]]] = {
    SoilType.ALLUVIAL: {
        "description": "Formed by river deposits, rich in nutrients",
        "suitable_crops": "Rice, wheat, sugarcane, vegetables",
        "characteristics": "Sandy to loamy texture, good drainage",
    },
    SoilType.BLACK: {
        "description": "Rich in clay, retains moisture well",
        "suitable_crops": "Cotton, soybean, wheat, sunflower",
        "characteristics": "High water retention, cracks when dry",
    },
    SoilType.CLAY: {
        "description": "Fine particles, holds nutrients well",
        "suitable_crops": "Rice, wheat (with drainage management)",
        "characteristics": "Poor drainage, sticky when wet",
    },
    SoilType.RED: {
        "description": "Rich in iron oxides, well-drained",
        "suitable_crops": "Groundnut, millets, pulses, potato",
        "characteristics": "Acidic, requires lime and organic matter",
    },
}
