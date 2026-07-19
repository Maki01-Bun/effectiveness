from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load("random_forest_subsidy.pkl")

class PredictionInput(BaseModel):
    subsidy_type: int
    farm_size: float
    crop_yield_before: float
    crop_yield_after: float
    income_before: float
    income_after: float
    pest: int
    calamity: int

@app.get("/")
def home():
    return {"message": "AgriSubsidy ML API Running"}

@app.post("/predict")
def predict(data: PredictionInput):

    features = [[
        data.subsidy_type,
        data.farm_size,
        data.crop_yield_before,
        data.crop_yield_after,
        data.income_before,
        data.income_after,
        data.pest,
        data.calamity
    ]]

    prediction = model.predict(features)

    return {
        "effectiveness": str(prediction[0])
    }