from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import joblib

# Create FastAPI app
app = FastAPI()

# Load models
soil_type_model = tf.keras.models.load_model('model.h5')
fertility_model = joblib.load('random_forest_model.joblib')

# Class dictionary for soil types
class_dict = {
    "Alluvial soil": 0,
    "Black Soil": 1,
    "Clay soil": 2,
    "Red soil": 3
}
class_list = list(class_dict.keys())

# Helper function to predict soil type
def predict_soil_type(img):
    resized_img = img.resize((299, 299))
    img_array = np.asarray(resized_img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    predictions = soil_type_model.predict(img_array)[0]

    max_index = np.argmax(predictions)
    predicted_label = class_list[max_index]
    predicted_probability = predictions[max_index]

    return predicted_label, predicted_probability

# Helper function to preprocess soil data
def preprocess_soil_data(input_data: dict):
    input_data_float = {k: float(v) for k, v in input_data.items()}
    transformed = {k: np.log10(v + 1e-10) for k, v in input_data_float.items()}
    return np.array(list(transformed.values())).reshape(1, -1)

# Main endpoint
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    N: float = Form(...),
    P: float = Form(...),
    K: float = Form(...),
    pH: float = Form(...),
    EC: float = Form(...),
    OC: float = Form(...),
    S: float = Form(...),
    Zn: float = Form(...),
    Fe: float = Form(...),
    Cu: float = Form(...),
    Mn: float = Form(...),
    B: float = Form(...)
):
    try:
        # Read and predict soil type from image
        img = Image.open(io.BytesIO(await file.read()))
        predicted_label, predicted_probability = predict_soil_type(img)

        # Predict soil fertility
        soil_data = {
            "N": N, "P": P, "K": K, "pH": pH, "EC": EC,
            "OC": OC, "S": S, "Zn": Zn, "Fe": Fe,
            "Cu": Cu, "Mn": Mn, "B": B
        }
        preprocessed_data = preprocess_soil_data(soil_data)
        fertility_prediction = fertility_model.predict(preprocessed_data)

        # Interpret fertility prediction
        fertility_status = "Suitable for cultivation" if fertility_prediction[0] == 1 else "Not suitable for cultivation"

        # Return all predictions
        return {
            "soil_type": predicted_label,
            "soil_type_probability": round(float(predicted_probability), 2),
            "fertility": fertility_status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
