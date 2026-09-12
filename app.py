from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()


# ============================================================
# LOAD MODEL
# ============================================================

try:
    model = joblib.load("random_forest_subsidy.pkl")

    print("=" * 60)
    print("MODEL LOADED")
    print("=" * 60)

    # Get the Random Forest from the pipeline
    if hasattr(model, "named_steps"):
        if "classifier" in model.named_steps:
            rf_model = model.named_steps["classifier"]
        else:
            rf_model = model.steps[-1][1]
    else:
        rf_model = model

    # Get EXACT feature order used during training
    if hasattr(rf_model, "feature_names_in_"):
        MODEL_FEATURES = list(rf_model.feature_names_in_)
    else:
        MODEL_FEATURES = None

    print("Features used by model:")

    if MODEL_FEATURES:
        for i, feature in enumerate(MODEL_FEATURES, 1):
            print(f"{i}. {feature}")

    MODEL_LOADED = True

except Exception as e:

    print("MODEL LOADING ERROR:")
    print(e)

    model = None
    rf_model = None
    MODEL_FEATURES = None
    MODEL_LOADED = False


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "status": "running",
        "model_loaded": MODEL_LOADED,
        "model_features": MODEL_FEATURES
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "model_loaded": MODEL_LOADED,
        "model_features": MODEL_FEATURES
    }


# ============================================================
# INPUT
# ============================================================

class PredictionInput(BaseModel):

    farm_size: float
    average_yield: float
    crop_yield_after: float
    selling_price: float
    subsidy_received: float
    feedback_score: float


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
def predict(data: PredictionInput):

    if model is None:

        return {
            "success": False,
            "error": "Model not loaded"
        }

    if MODEL_FEATURES is None:

        return {
            "success": False,
            "error": "Could not determine model feature order"
        }

    # ========================================================
    # INPUT VALUES
    # ========================================================

    input_values = {

        "Farm Size (ha)": data.farm_size,

        "Average Yield (bags/ha)": data.average_yield,

        "Crop Yield (bags/ha)": data.crop_yield_after,

        "Average Selling Price (₱/kg)": data.selling_price,

        "Subsidy Received": data.subsidy_received,

        "Feedback Score": data.feedback_score

    }

    # ========================================================
    # CHECK MISSING FEATURES
    # ========================================================

    missing_features = [
        feature
        for feature in MODEL_FEATURES
        if feature not in input_values
    ]

    if missing_features:

        return {
            "success": False,
            "error": "Missing features",
            "missing_features": missing_features,
            "model_features": MODEL_FEATURES
        }

    # ========================================================
    # BUILD DATAFRAME USING EXACT MODEL ORDER
    # ========================================================

    df = pd.DataFrame(
        [[input_values[feature] for feature in MODEL_FEATURES]],
        columns=MODEL_FEATURES
    )

    # ========================================================
    # DEBUG
    # ========================================================

    print("=" * 60)
    print("PREDICTION")
    print("=" * 60)

    print(df)

    print("\nFEATURE ORDER:")

    for i, feature in enumerate(df.columns, 1):

        print(f"{i}. {feature}")

    # ========================================================
    # PREDICT
    # ========================================================

    try:

        prediction = model.predict(df)[0]

        # ====================================================
        # EFFECTIVENESS LABEL
        # ====================================================

        LABEL_NAMES = {
            0: "Not Effective",
            1: "Moderately Effective",
            2: "Effective"
        }

        prediction_code = int(prediction)

        effectiveness = LABEL_NAMES.get(
            prediction_code,
            str(prediction)
        )

        # ====================================================
        # YIELD COMPARISON
        # ====================================================

        average_yield = float(data.average_yield)

        crop_yield = float(data.crop_yield_after)

        yield_change = crop_yield - average_yield

        if yield_change > 0:

            yield_status = "Increased"

        elif yield_change < 0:

            yield_status = "Decreased"

        else:

            yield_status = "No Change"

        # ====================================================
        # PERCENTAGE CHANGE
        # ====================================================

        if average_yield != 0:

            yield_change_percent = (
                yield_change / average_yield
            ) * 100

        else:

            yield_change_percent = 0

        # ====================================================
        # PROBABILITIES
        # ====================================================

        probabilities = {}

        if hasattr(model, "predict_proba"):

            probability_values = model.predict_proba(df)[0]

            for class_value, probability in zip(
                model.classes_,
                probability_values
            ):

                class_code = int(class_value)

                class_name = LABEL_NAMES.get(
                    class_code,
                    str(class_code)
                )

                probabilities[class_name] = round(
                    float(probability) * 100,
                    2
                )

        # ====================================================
        # RESPONSE
        # ====================================================

        return {

            "success": True,

            "effectiveness": effectiveness,

            "effectiveness_code": prediction_code,

            "average_yield": round(
                average_yield,
                2
            ),

            "crop_yield": round(
                crop_yield,
                2
            ),

            "yield_change": round(
                yield_change,
                2
            ),

            "yield_change_percent": round(
                yield_change_percent,
                2
            ),

            "yield_status": yield_status,

            "probabilities": probabilities,

            "model_features": MODEL_FEATURES

        }

    except Exception as e:

        print("=" * 60)
        print("PREDICTION ERROR")
        print("=" * 60)
        print(e)

        return {

            "success": False,

            "error": str(e),

            "model_features": MODEL_FEATURES,

            "received_features": list(df.columns)

        }