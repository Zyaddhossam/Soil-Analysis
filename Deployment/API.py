from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
from PIL import Image
import io
import tensorflow as tf

# Create FastAPI app
app = FastAPI()

# Class dictionary for soil types
class_dict = {
    "Alluvial soil": 0,
    "Black Soil": 1,
    "Clay soil": 2,
    "Red soil": 3
}

# Load the pre-trained model
model = tf.keras.models.load_model('model.h5')

# Helper function to predict image label
def predict_image(img):
    labels = list(class_dict.keys())
    resized_img = img.resize((299, 299))
    img_array = np.asarray(resized_img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    predictions = model.predict(img_array)[0]

    max_index = np.argmax(predictions)
    predicted_label = labels[max_index]
    predicted_probability = predictions[max_index]

    return predicted_label, predicted_probability

# Request model for predict response
class PredictionResponse(BaseModel):
    predicted_label: str
    predicted_probability: float

# Define the prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    try:
        # Read the image file
        img = Image.open(io.BytesIO(await file.read()))
        predicted_label, predicted_probability = predict_image(img)

        # Return prediction response
        return PredictionResponse(
            predicted_label=predicted_label,
            predicted_probability=round(float(predicted_probability), 2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
