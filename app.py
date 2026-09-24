from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pathlib import Path
import joblib
import pandas as pd
import numpy as np


# ============================================================
# AGRISUBSIDY EFFECTIVENESS PREDICTION API
# ============================================================
#
# MODEL:
# Random Forest Classifier
#
# TARGET:
# 0 = Not Effective
# 1 = Moderately Effective
# 2 = Effective
#
# IMPORTANT:
# This API must reproduce the same feature engineering
# that was used during model training.
#
# ============================================================


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="AgriSubsidy Effectiveness API",
    description="Overall Farmer Effectiveness Prediction",
    version="3.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
# MODEL LOCATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR /
    "random_forest_subsidy.pkl"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = None
rf_model = None
MODEL_FEATURES = None
MODEL_LOADED = False


try:

    model = joblib.load(
        MODEL_PATH
    )

    print("=" * 70)
    print("AGRISUBSIDY MODEL LOADED")
    print("=" * 70)

    print(
        f"Model path: {MODEL_PATH}"
    )

    # --------------------------------------------------------
    # Get Random Forest from Pipeline
    # --------------------------------------------------------

    if hasattr(
        model,
        "named_steps"
    ):

        if "classifier" in model.named_steps:

            rf_model = (
                model.named_steps[
                    "classifier"
                ]
            )

        else:

            rf_model = (
                model.steps[-1][1]
            )

    else:

        rf_model = model


    # --------------------------------------------------------
    # Get EXACT FEATURES used during training
    # --------------------------------------------------------

    if hasattr(
        rf_model,
        "feature_names_in_"
    ):

        MODEL_FEATURES = list(
            rf_model.feature_names_in_
        )

    elif hasattr(
        model,
        "feature_names_in_"
    ):

        MODEL_FEATURES = list(
            model.feature_names_in_
        )

    else:

        MODEL_FEATURES = None


    print("\nMODEL FEATURES:")

    if MODEL_FEATURES:

        for index, feature in enumerate(
            MODEL_FEATURES,
            1
        ):

            print(
                f"{index}. {feature}"
            )

    else:

        print(
            "WARNING: Model feature names could not be determined."
        )


    MODEL_LOADED = True

    print(
        "\nModel loaded successfully."
    )

    print("=" * 70)


except Exception as e:

    print("=" * 70)
    print("MODEL LOADING ERROR")
    print("=" * 70)

    print(
        str(e)
    )

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

        "model_loaded":
            MODEL_LOADED,

        "model_features":
            MODEL_FEATURES,

        "prediction_endpoint":
            "/predict"

    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {

        "status": "ok",

        "model_loaded":
            MODEL_LOADED,

        "model_features":
            MODEL_FEATURES

    }


# ============================================================
# PREDICTION INPUT
# ============================================================

class PredictionInput(BaseModel):

    # --------------------------------------------------------
    # FARM INFORMATION
    # --------------------------------------------------------

    farm_size: float = Field(
        ...,
        gt=0
    )

    # --------------------------------------------------------
    # PLANTING TYPE
    #
    # 0 = Hybrid
    # 1 = Inbred
    #
    # NOTE:
    # Planting Type is accepted by the API but is NOT directly
    # passed to the model because it was not included in the
    # training FEATURES list.
    # --------------------------------------------------------

    planting_type: int = Field(
        ...,
        ge=0,
        le=1
    )

    # --------------------------------------------------------
    # YIELD INFORMATION
    # --------------------------------------------------------

    average_yield: float = Field(
        ...,
        gt=0
    )

    crop_yield_after: float = Field(
        ...,
        ge=0
    )

    # --------------------------------------------------------
    # MARKET / SUBSIDY
    # --------------------------------------------------------

    selling_price: float = Field(
        ...,
        ge=0
    )

    subsidy_received: float = Field(
        ...,
        ge=0
    )

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

    farm_size = float(
        data.farm_size
    )

    average_yield = float(
        data.average_yield
    )

    crop_yield = float(
        data.crop_yield_after
    )

    selling_price = float(
        data.selling_price
    )

    subsidy_received = float(
        data.subsidy_received
    )


    # ========================================================
    # PLANTING TYPE INFORMATION
    # ========================================================
    #
    # 0 = Hybrid
    # 1 = Inbred
    #
    # Expected average yield:
    #
    # Hybrid  = 5.5 tons/ha
    # Inbred  = 6.0 tons/ha
    #
    # This value is used for reference/validation only.
    #
    # The actual "Average Yield (tons/ha)" sent to the model
    # remains the value supplied by the survey/database.
    #
    # ========================================================

    if data.planting_type == 0:

        planting_type_name = "Hybrid"

        planting_type_base_yield = 5.5

    else:

        planting_type_name = "Inbred"

        planting_type_base_yield = 6.0


    calculated_expected_yield = (
        farm_size *
        planting_type_base_yield
    )


    # ========================================================
    # Q1-Q10
    # ========================================================

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


    # ========================================================
    # FEATURE ENGINEERING
    # ========================================================
    #
    # THESE MUST MATCH THE TRAINING CODE.
    #
    # ========================================================


    # --------------------------------------------------------
    # Q MEAN
    # --------------------------------------------------------

    q_mean = np.mean(
        feedback_values
    )


    # --------------------------------------------------------
    # Q TOTAL
    # --------------------------------------------------------

    q_total = np.sum(
        feedback_values
    )


    # --------------------------------------------------------
    # Q STANDARD DEVIATION
    #
    # pandas .std() uses sample standard deviation
    # (ddof=1), so we reproduce that behavior.
    # --------------------------------------------------------

    q_std = np.std(
        feedback_values,
        ddof=1
    )


    # --------------------------------------------------------
    # Q MIN
    # --------------------------------------------------------

    q_min = np.min(
        feedback_values
    )


    # --------------------------------------------------------
    # Q MAX
    # --------------------------------------------------------

    q_max = np.max(
        feedback_values
    )


    # ========================================================
    # YIELD FEATURES
    # ========================================================


    # --------------------------------------------------------
    # EXPECTED YIELD
    #
    # Training:
    #
    # Expected_Yield =
    # Farm Size * Average Yield
    # --------------------------------------------------------

    expected_yield = (
        farm_size *
        average_yield
    )


    # --------------------------------------------------------
    # YIELD DIFFERENCE
    #
    # Training:
    #
    # Crop Yield - Expected Yield
    # --------------------------------------------------------

    yield_difference = (

        crop_yield
        -
        expected_yield

    )


    # --------------------------------------------------------
    # ACTUAL YIELD PER HECTARE
    # --------------------------------------------------------

    if farm_size > 0:

        actual_yield_per_ha = (

            crop_yield /
            farm_size

        )

    else:

        actual_yield_per_ha = 0


    # --------------------------------------------------------
    # YIELD EFFICIENCY
    # --------------------------------------------------------

    if expected_yield > 0:

        yield_efficiency = (

            crop_yield /
            expected_yield

        )

    else:

        yield_efficiency = 0


    # ========================================================
    # SUBSIDY FEATURE
    # ========================================================

    if farm_size > 0:

        subsidy_per_ha = (

            subsidy_received /
            farm_size

        )

    else:

        subsidy_per_ha = 0


    # ========================================================
    # BUILD ALL POSSIBLE FEATURES
    # ========================================================
    #
    # Planting Type is included here only so the API can
    # receive it, but if it is not in MODEL_FEATURES it will
    # NOT be sent to the Random Forest.
    #
    # ========================================================

    input_values = {

        # ----------------------------------------------------
        # Original features
        # ----------------------------------------------------

        "Farm Size (ha)":
            farm_size,

        "Average Yield (tons/ha)":
            average_yield,

        "Crop Yield (tons)":
            crop_yield,

        "Average Selling Price (₱/kg)":
            selling_price,

        "Subsidy Received":
            subsidy_received,

        # ----------------------------------------------------
        # Q1-Q10
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Q aggregates
        # ----------------------------------------------------

        "Q_Mean":
            q_mean,

        "Q_Total":
            q_total,

        "Q_Std":
            q_std,

        "Q_Min":
            q_min,

        "Q_Max":
            q_max,

        # ----------------------------------------------------
        # Yield engineered features
        # ----------------------------------------------------

        "Expected_Yield":
            expected_yield,

        "Yield_Difference":
            yield_difference,

        "Actual_Yield_Per_Ha":
            actual_yield_per_ha,

        "Yield_Efficiency":
            yield_efficiency,

        # ----------------------------------------------------
        # Subsidy engineered feature
        # ----------------------------------------------------

        "Subsidy_per_Ha":
            subsidy_per_ha,

        # ----------------------------------------------------
        # Planting type
        #
        # Not part of the current trained feature list.
        # ----------------------------------------------------

        "Planting Type":
            data.planting_type

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
                "Missing features required by trained model",

            "missing_features":
                missing_features,

            "model_features":
                MODEL_FEATURES

        }


    # ========================================================
    # BUILD DATAFRAME
    #
    # IMPORTANT:
    # ONLY FEATURES ACTUALLY USED DURING TRAINING
    # ARE SENT TO THE MODEL.
    #
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
    # CHECK NUMERIC DATA
    # ========================================================

    df = df.apply(
        pd.to_numeric,
        errors="coerce"
    )


    if df.isnull().any().any():

        bad_columns = list(
            df.columns[
                df.isnull().any()
            ]
        )

        return {

            "success": False,

            "error":
                "Invalid numeric feature values",

            "invalid_features":
                bad_columns

        }


    # ========================================================
    # DEBUG OUTPUT
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "OVERALL FARMER EFFECTIVENESS PREDICTION"
    )
    print("=" * 70)

    print("\nINPUT VALUES:")

    print(
        f"Farm Size:             {farm_size}"
    )

    print(
        f"Planting Type:         {planting_type_name}"
    )

    print(
        f"Average Yield:         {average_yield}"
    )

    print(
        f"Crop Yield:            {crop_yield}"
    )

    print(
        f"Selling Price:         {selling_price}"
    )

    print(
        f"Subsidy Received:      {subsidy_received}"
    )


    print("\nENGINEERED FEATURES:")

    print(
        f"Q Mean:                {q_mean:.4f}"
    )

    print(
        f"Q Total:               {q_total:.4f}"
    )

    print(
        f"Q Std:                 {q_std:.4f}"
    )

    print(
        f"Q Min:                 {q_min:.4f}"
    )

    print(
        f"Q Max:                 {q_max:.4f}"
    )

    print(
        f"Expected Yield:        {expected_yield:.4f}"
    )

    print(
        f"Yield Difference:      {yield_difference:.4f}"
    )

    print(
        f"Actual Yield/Ha:       {actual_yield_per_ha:.4f}"
    )

    print(
        f"Yield Efficiency:      {yield_efficiency:.4f}"
    )

    print(
        f"Subsidy/Ha:            {subsidy_per_ha:.4f}"
    )


    print("\nMODEL FEATURE ORDER:")

    for index, feature in enumerate(
        df.columns,
        1
    ):

        print(
            f"{index}. {feature} = {df.iloc[0][feature]}"
        )


    # ========================================================
    # PREDICT
    # ========================================================

    try:

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = model.predict(
            df
        )[0]


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

        yield_change = (

            crop_yield
            -
            average_yield

        )


        if yield_change > 0:

            yield_status = "Increased"

        elif yield_change < 0:

            yield_status = "Decreased"

        else:

            yield_status = "No Change"


        # ----------------------------------------------------
        # Percentage change
        # ----------------------------------------------------

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


            # ------------------------------------------------
            # Get classes
            # ------------------------------------------------

            if hasattr(
                model,
                "classes_"
            ):

                classes = (
                    model.classes_
                )

            elif hasattr(
                rf_model,
                "classes_"
            ):

                classes = (
                    rf_model.classes_
                )

            else:

                classes = []


            for class_value, probability in zip(

                classes,

                probability_values

            ):

                class_code = int(
                    class_value
                )


                class_name = (
                    LABEL_NAMES.get(
                        class_code,
                        str(class_code)
                    )
                )


                probabilities[
                    class_name
                ] = round(

                    float(
                        probability
                    ) * 100,

                    2

                )


        # ====================================================
        # FEEDBACK SUMMARY
        # ====================================================

        feedback_average = (

            sum(
                feedback_values
            )
            /
            len(
                feedback_values
            )

        )


        feedback_total = sum(
            feedback_values
        )


        # ====================================================
        # RESPONSE
        # ====================================================

        return {

            "success": True,

            "evaluation_type":
                "Overall Farmer Effectiveness",

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            "effectiveness":
                effectiveness,

            "effectiveness_code":
                prediction_code,

            # ------------------------------------------------
            # Planting type
            # ------------------------------------------------

            "planting_type":
                data.planting_type,

            "planting_type_name":
                planting_type_name,

            "planting_type_base_yield":
                planting_type_base_yield,

            "calculated_expected_yield":
                round(
                    calculated_expected_yield,
                    2
                ),

            # ------------------------------------------------
            # Yield
            # ------------------------------------------------

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

            "expected_yield":
                round(
                    expected_yield,
                    2
                ),

            "actual_yield_per_ha":
                round(
                    actual_yield_per_ha,
                    2
                ),

            "yield_efficiency":
                round(
                    yield_efficiency,
                    4
                ),

            "yield_difference":
                round(
                    yield_difference,
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

            # ------------------------------------------------
            # Feedback
            # ------------------------------------------------

            "feedback_average":
                round(
                    feedback_average,
                    2
                ),

            "feedback_total":
                int(
                    feedback_total
                ),

            # ------------------------------------------------
            # Subsidy
            # ------------------------------------------------

            "subsidy_received":
                round(
                    subsidy_received,
                    2
                ),

            "subsidy_per_ha":
                round(
                    subsidy_per_ha,
                    2
                ),

            # ------------------------------------------------
            # Probabilities
            # ------------------------------------------------

            "probabilities":
                probabilities,

            # ------------------------------------------------
            # Model information
            # ------------------------------------------------

            "model_features":
                MODEL_FEATURES

        }


    # ========================================================
    # PREDICTION ERROR
    # ========================================================

    except Exception as e:

        print("=" * 70)
        print("PREDICTION ERROR")
        print("=" * 70)

        print(
            str(e)
        )


        return {

            "success": False,

            "error":
                str(e),

            "model_features":
                MODEL_FEATURES,

            "received_features":
                list(df.columns)

        }