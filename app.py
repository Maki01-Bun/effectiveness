from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="AgriSubsidy Effectiveness API",
    description="Overall Farmer Effectiveness Prediction",
    version="2.0"
)


# ============================================================
# LABEL NAMES
# ============================================================

LABEL_NAMES = {
    0: "Not Effective",
    1: "Moderately Effective",
    2: "Effective"
}


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = joblib.load(
        "random_forest_subsidy.pkl"
    )

    print("=" * 60)
    print("MODEL LOADED")
    print("=" * 60)

    # --------------------------------------------------------
    # Get Random Forest from Pipeline
    # --------------------------------------------------------

    if hasattr(model, "named_steps"):

        if "classifier" in model.named_steps:

            rf_model = model.named_steps["classifier"]

        else:

            rf_model = model.steps[-1][1]

    else:

        rf_model = model


    # --------------------------------------------------------
    # Get EXACT feature order used during training
    # --------------------------------------------------------

    if hasattr(
        rf_model,
        "feature_names_in_"
    ):

        MODEL_FEATURES = list(
            rf_model.feature_names_in_
        )

    else:

        MODEL_FEATURES = None


    print("Features used by model:")

    if MODEL_FEATURES:

        for i, feature in enumerate(
            MODEL_FEATURES,
            1
        ):

            print(
                f"{i}. {feature}"
            )


    MODEL_LOADED = True


except Exception as e:

    print("=" * 60)
    print("MODEL LOADING ERROR")
    print("=" * 60)

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

        "model_features": MODEL_FEATURES,

        "prediction_endpoint": "/predict"

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
# PREDICTION INPUT
# ============================================================

class PredictionInput(BaseModel):

    # --------------------------------------------------------
    # FARM INFORMATION
    # --------------------------------------------------------

    farm_size: float

    planting_type: int

    # --------------------------------------------------------
    # YIELD INFORMATION
    # --------------------------------------------------------

    average_yield: float

    crop_yield_after: float

    # --------------------------------------------------------
    # MARKET / SUBSIDY
    # --------------------------------------------------------

    selling_price: float

    subsidy_received: float

    # --------------------------------------------------------
    # FEEDBACK QUESTIONS
    # --------------------------------------------------------

    q1: int = Field(
        ...,
        ge=1,
        le=5
    )

    q2: int = Field(
        ...,
        ge=1,
        le=5
    )

    q3: int = Field(
        ...,
        ge=1,
        le=5
    )

    q4: int = Field(
        ...,
        ge=1,
        le=5
    )

    q5: int = Field(
        ...,
        ge=1,
        le=5
    )

    q6: int = Field(
        ...,
        ge=1,
        le=5
    )

    q7: int = Field(
        ...,
        ge=1,
        le=5
    )

    q8: int = Field(
        ...,
        ge=1,
        le=5
    )

    q9: int = Field(
        ...,
        ge=1,
        le=5
    )

    q10: int = Field(
        ...,
        ge=1,
        le=5
    )


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
def predict(
    data: PredictionInput
):

    # ========================================================
    # CHECK MODEL
    # ========================================================

    if model is None:

        return {

            "success": False,

            "error":
                "Model not loaded"

        }


    # ========================================================
    # CHECK MODEL FEATURES
    # ========================================================

    if MODEL_FEATURES is None:

        return {

            "success": False,

            "error":
                "Could not determine model feature order"

        }


    # ========================================================
    # INPUT VALUES
    # ========================================================

    input_values = {

        "Farm Size (ha)":
            data.farm_size,

        "Planting Type":
            data.planting_type,

        "Average Yield (tons/ha)":
            data.average_yield,

        "Crop Yield (tons)":
            data.crop_yield_after,

        "Average Selling Price (₱/kg)":
            data.selling_price,

        "Q1":
            data.q1,

        "Q2":
            data.q2,

        "Q3":
            data.q3,

        "Q4":
            data.q4,

        "Q5":
            data.q5,

        "Q6":
            data.q6,

        "Q7":
            data.q7,

        "Q8":
            data.q8,

        "Q9":
            data.q9,

        "Q10":
            data.q10,

        "Subsidy Received":
            data.subsidy_received

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

            "error":
                "Missing features",

            "missing_features":
                missing_features,

            "model_features":
                MODEL_FEATURES

        }


    # ========================================================
    # BUILD DATAFRAME
    #
    # IMPORTANT:
    # Uses EXACT feature order from trained model.
    # ========================================================

    df = pd.DataFrame(

        [
            [
                input_values[feature]

                for feature in MODEL_FEATURES

            ]
        ],

        columns=MODEL_FEATURES

    )


    # ========================================================
    # DEBUG
    # ========================================================

    print("=" * 60)
    print("OVERALL FARMER EFFECTIVENESS PREDICTION")
    print("=" * 60)

    print("\nINPUT DATA:")

    print(df)


    print("\nFEATURE ORDER:")

    for i, feature in enumerate(
        df.columns,
        1
    ):

        print(
            f"{i}. {feature}"
        )


    # ========================================================
    # PREDICT
    # ========================================================

    try:

        prediction = model.predict(df)[0]


        # ----------------------------------------------------
        # EFFECTIVENESS CODE
        # ----------------------------------------------------

        prediction_code = int(
            prediction
        )


        effectiveness = LABEL_NAMES.get(

            prediction_code,

            str(prediction)

        )


        # ====================================================
        # YIELD COMPARISON
        # ====================================================

        average_yield = float(
            data.average_yield
        )

        crop_yield = float(
            data.crop_yield_after
        )


        yield_change = (

            crop_yield -
            average_yield

        )


        if yield_change > 0:

            yield_status = "Increased"

        elif yield_change < 0:

            yield_status = "Decreased"

        else:

            yield_status = "No Change"


        # ====================================================
        # YIELD PERCENTAGE CHANGE
        # ====================================================

        if average_yield != 0:

            yield_change_percent = (

                yield_change /
                average_yield

            ) * 100

        else:

            yield_change_percent = 0


        # ====================================================
        # MODEL PROBABILITIES
        # ====================================================

        probabilities = {}


        if hasattr(
            model,
            "predict_proba"
        ):

            probability_values = (
                model.predict_proba(df)[0]
            )


            # ----------------------------------------------
            # Pipeline-safe class extraction
            # ----------------------------------------------

            if hasattr(
                model,
                "classes_"
            ):

                classes = model.classes_

            elif hasattr(
                rf_model,
                "classes_"
            ):

                classes = rf_model.classes_

            else:

                classes = []


            for class_value, probability in zip(

                classes,

                probability_values

            ):

                class_code = int(
                    class_value
                )


                class_name = LABEL_NAMES.get(

                    class_code,

                    str(class_code)

                )


                probabilities[class_name] = round(

                    float(probability) * 100,

                    2

                )


        # ====================================================
        # FEEDBACK SUMMARY
        # ====================================================

        feedback_values = [

            data.q1,
            data.q2,
            data.q3,
            data.q4,
            data.q5,
            data.q6,
            data.q7,
            data.q8,
            data.q9,
            data.q10

        ]


        feedback_average = (

            sum(feedback_values) /
            len(feedback_values)

        )


        # ====================================================
        # RESPONSE
        # ====================================================

        return {

            "success": True,

            "evaluation_type":
                "Overall Farmer Effectiveness",

            "effectiveness":
                effectiveness,

            "effectiveness_code":
                prediction_code,

            "average_yield":
                round(
                    average_yield,
                    2
                ),

            "crop_yield":
                round(
                    crop_yield,
                    2
                ),

            "yield_change":
                round(
                    yield_change,
                    2
                ),

            "yield_change_percent":
                round(
                    yield_change_percent,
                    2
                ),

            "yield_status":
                yield_status,

            "feedback_average":
                round(
                    feedback_average,
                    2
                ),

            "probabilities":
                probabilities,

            "model_features":
                MODEL_FEATURES

        }


    # ========================================================
    # PREDICTION ERROR
    # ========================================================

    except Exception as e:

        print("=" * 60)
        print("PREDICTION ERROR")
        print("=" * 60)

        print(e)


        return {

            "success": False,

            "error":
                str(e),

            "model_features":
                MODEL_FEATURES,

            "received_features":
                list(df.columns)

        }