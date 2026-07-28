from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# ----------------------------
# Create FastAPI app
# ----------------------------
app = FastAPI(
    title="AgriSubsidy Prediction API",
    version="1.0"
)

# ----------------------------
# Load trained model
# ----------------------------
try:
    model = joblib.load("random_forest_subsidy.pkl")
    print("Model loaded successfully.")
except Exception as e:
    print("Unable to load model:", e)
    model = None

# ----------------------------
# Request Schema
# ----------------------------
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

# ----------------------------
# Home
# ----------------------------
@app.get("/")
def home():
    return {
        "message": "AgriSubsidy Random Forest API is running."
    }
# ----------------------------
# Prediction Endpoint
# ----------------------------
@app.post("/predict")
def predict(data: PredictionInput):

    if model is None:
        return {
            "error": "Model not loaded."
        }

    input_data = pd.DataFrame([{
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

    prediction = model.predict(input_data)[0]

    return {
        "effectiveness": prediction
    }