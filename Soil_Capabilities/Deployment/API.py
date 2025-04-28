
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib


model = joblib.load('random_forest_model.joblib')


class InputData(BaseModel):
    N: float
    P: float
    K: float
    pH: float
    EC: float
    OC: float
    S: float
    Zn: float
    Fe: float
    Cu: float
    Mn: float
    B: float

app = FastAPI()


def preprocess(input_data: dict):
    transformed = {k: np.log10(v + 1e-10) for k, v in input_data.items()}
    return np.array(list(transformed.values())).reshape(1, -1)


@app.post("/predict")
def predict(data: InputData):
 
    input_dict = data.dict()
    
    preprocessed_input = preprocess(input_dict)

    prediction = model.predict(preprocessed_input)
   
    return {"prediction": int(prediction[0])}
