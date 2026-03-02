# API Reference

Base URL: `http://localhost:8000/api/v1`

---

## Health Endpoints

### GET `/health`

Check API health and model loading status.

**Response** `200 OK`:

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "models": {
    "soil_classifier": true,
    "fertility_predictor": true
  }
}
```

### GET `/ready`

Check readiness (all models loaded).

**Response** `200 OK`:

```json
{ "status": "ready" }
```

---

## Prediction Endpoints

### GET `/predictions/model-info`

Get information about the loaded models.

**Response** `200 OK`:

```json
{
  "soil_classifier": {
    "backbone": "efficientnet_b0",
    "image_size": [224, 224],
    "num_classes": 4,
    "class_names": ["Alluvial Soil", "Black Soil", "Clay Soil", "Red Soil"],
    "loaded": true
  },
  "fertility_predictor": {
    "num_classes": 3,
    "class_names": ["Less Fertile", "Fertile", "Highly Fertile"],
    "feature_names": ["N", "P", "K", "pH", "EC", "OC", "S", "Zn", "Fe", "Cu", "Mn", "B"],
    "loaded": true
  }
}
```

---

### POST `/predictions/soil-type`

Classify soil type from an uploaded image.

**Request**: `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | File (JPEG/PNG) | Yes | Soil image |

**Query Parameters**:

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `include_probabilities` | bool | false | Include all class probabilities |

**Response** `200 OK`:

```json
{
  "class_id": 0,
  "class_name": "Alluvial Soil",
  "confidence": 0.9234,
  "backbone": "efficientnet_b0",
  "characteristics": {
    "description": "Formed by river deposits, rich in nutrients",
    "suitable_crops": "Rice, wheat, sugarcane, vegetables",
    "characteristics": "Sandy to loamy texture, good drainage"
  },
  "probabilities": {
    "Alluvial Soil": 0.9234,
    "Black Soil": 0.0412,
    "Clay Soil": 0.0201,
    "Red Soil": 0.0153
  }
}
```

**Errors**:

| Code | Cause |
|------|-------|
| 400 | Invalid file type (not JPEG/PNG) |
| 503 | Model not loaded |
| 500 | Prediction failure |

---

### POST `/predictions/fertility`

Predict soil fertility from nutrient lab values.

**Request**: `application/json`

```json
{
  "N": 280, "P": 45, "K": 320,
  "pH": 6.5, "EC": 0.45, "OC": 0.75,
  "S": 12, "Zn": 1.2, "Fe": 8.5,
  "Cu": 1.8, "Mn": 15, "B": 0.5
}
```

All 12 nutrient fields are required (float ≥ 0; pH ≤ 14).

**Response** `200 OK`:

```json
{
  "class_id": 2,
  "class_name": "Highly Fertile",
  "confidence": 0.87,
  "recommendation": "Excellent soil fertility! ...",
  "needs_lab_test": false,
  "probabilities": {
    "Less Fertile": 0.05,
    "Fertile": 0.08,
    "Highly Fertile": 0.87
  },
  "warnings": []
}
```

---

### POST `/predictions/analyze`

Combined analysis: soil type from image AND fertility from nutrients.

**Request**: `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | File | Yes | Soil image |
| `N` | float | Yes | Nitrogen (kg/ha) |
| `P` | float | Yes | Phosphorus (kg/ha) |
| ... | ... | ... | (all 12 nutrient fields) |

**Response** `200 OK`:

```json
{
  "soil_type": { ... },
  "fertility": { ... }
}
```

Both sub-objects follow the schemas above.

---

## Error Responses

All errors use the standard format:

```json
{
  "detail": "Human-readable error message"
}
```

| Code | Meaning |
|------|---------|
| 400 | Bad request (invalid input) |
| 422 | Validation error (missing/invalid fields) |
| 500 | Internal server error |
| 503 | Service unavailable (model not loaded) |
