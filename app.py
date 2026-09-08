from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

try:
    model = joblib.load("random_forest_subsidy.pkl")
    print("Model classes:", model.classes_)
    MODEL_LOADED = True
except Exception as e:
    print(e)
    MODEL_LOADED = False
    model = None


@app.get("/")
def home():
    return {
        "status": "running",
        "model_loaded": MODEL_LOADED
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL_LOADED
    }


class PredictionInput(BaseModel):
    farm_size: float
    average_yield: float
    crop_yield_after: float
    selling_price: float
    subsidy_received: str
    feedback_score: float


@app.post("/predict")
def predict(data: PredictionInput):
    

    if model is None:
        return {"error": "Model not loaded"}

    df = pd.DataFrame([{
        "Farm Size (ha)": data.farm_size,
        "Average Yield (bags/ha)": data.average_yield,
        "Crop Yield (bags/ha)": data.crop_yield_after,
        "Average Selling Price (₱/kg)": data.selling_price,
        "Subsidy Received": data.subsidy_received,  
        "Feedback Score": data.feedback_score,
    }])

    prediction = model.predict(df)[0]

    return {
        "effectiveness": str(prediction)
    }