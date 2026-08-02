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
    subsidy_type: str
    farm_size: float
    crop_yield_before: float
    crop_yield_after: float
    income_before: float
    income_after: float
    feedback_score: float
    pest: str
    calamity: str


@app.post("/predict")
def predict(data: PredictionInput):
    

    if model is None:
        return {"error": "Model not loaded"}

    df = pd.DataFrame([{
        "Subsidy Type": data.subsidy_type,
        "Farm Size (ha)": data.farm_size,
        "Crop Yield Before": data.crop_yield_before,
        "Crop Yield After": data.crop_yield_after,
        "Income Before": data.income_before,
        "Income After": data.income_after,
        "Feedback Score": data.feedback_score,
        "Pest": data.pest,
        "Calamity": data.calamity
    }])

    prediction = model.predict(df)[0]

    return {
        "effectiveness": str(prediction)
    }