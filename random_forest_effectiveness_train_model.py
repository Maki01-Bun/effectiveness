# ============================================================
# AGRISUBSIDY EFFECTIVENESS PREDICTION
# RANDOM FOREST + SMOTE
# UPDATED DATASET - TONS
# ============================================================

import os
import random
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    learning_curve
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


# ============================================================
# SETTINGS
# ============================================================

random.seed(42)
np.random.seed(42)

warnings.filterwarnings("ignore")


# ============================================================
# UPDATED DATASET
# ============================================================

TRAIN_FILE = "datasets/subsidy_dataset_ml_ready.xlsx"

# The uploaded dataset contains Sheet1
SHEET_NAME = "Sheet1"

TARGET = "Effectiveness Label"

MODEL_FILE = "random_forest_subsidy.pkl"
FEATURES_FILE = "model_features.pkl"

CONFUSION_MATRIX_FILE = "confusion_matrix_test.png"
LEARNING_CURVE_FILE = "learning_curve.png"

FEATURE_IMPORTANCE_FILE = "feature_importance.xlsx"
TRAINING_METRICS_FILE = "training_metrics.xlsx"

RANDOM_STATE = 42


# ============================================================
# LABEL NAMES
# ============================================================

LABEL_NAMES = {
    0: "Not Effective",
    1: "Moderately Effective",
    2: "Effective"
}


# ============================================================
# MODEL FEATURES
# ============================================================

REQUIRED_FEATURES = [
    "Farm Size (ha)",
    "Average Yield (tons/ha)",
    "Crop Yield (tons)",
    "Average Selling Price (₱/kg)",
    "Feedback Score",
    "Subsidy Received"
]


# ============================================================
# HELPER
# ============================================================

def print_section(title):

    print("\n")
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# 1. LOAD DATASET
# ============================================================

print_section("1. LOADING UPDATED DATASET")


if not os.path.exists(TRAIN_FILE):

    raise FileNotFoundError(
        f"""
Dataset not found:

{TRAIN_FILE}

Make sure the updated Excel file is inside the datasets folder.
"""
    )


df = pd.read_excel(
    TRAIN_FILE,
    sheet_name=SHEET_NAME
)


print("Dataset loaded successfully.")

print(f"File   : {TRAIN_FILE}")
print(f"Sheet  : {SHEET_NAME}")
print(f"Rows   : {df.shape[0]}")
print(f"Columns: {df.shape[1]}")


print("\nColumns:")

for column in df.columns:

    print(f" - {column}")


# ============================================================
# 2. DATA VALIDATION
# ============================================================

print_section("2. DATA VALIDATION")


if TARGET not in df.columns:

    raise ValueError(
        f"Target column '{TARGET}' was not found."
    )


missing_values = df.isnull().sum()


print("\nMissing values:")

print(missing_values)


if missing_values.sum() > 0:

    raise ValueError(
        "The dataset contains missing values. "
        "Please clean the dataset before training."
    )


# ============================================================
# 3. FEATURE VALIDATION
# ============================================================

print_section("3. FEATURE VALIDATION")


missing_features = [
    feature
    for feature in REQUIRED_FEATURES
    if feature not in df.columns
]


if missing_features:

    raise ValueError(
        "The following required features are missing:\n"
        +
        "\n".join(
            f" - {feature}"
            for feature in missing_features
        )
    )


print("All required features were found.")


# ============================================================
# 4. TARGET VALIDATION
# ============================================================

print_section("4. EFFECTIVENESS LABEL VALIDATION")


if not pd.api.types.is_numeric_dtype(
    df[TARGET]
):

    raise TypeError(
        f"""
'{TARGET}' must contain numerical values.

Expected:
0 = Not Effective
1 = Moderately Effective
2 = Effective
"""
    )


df[TARGET] = df[TARGET].astype(int)


unique_labels = sorted(
    df[TARGET].unique().tolist()
)


print("Effectiveness Label values:")

print(unique_labels)


expected_labels = {0, 1, 2}


if set(unique_labels) != expected_labels:

    raise ValueError(
        f"""
The dataset must contain exactly:

0 = Not Effective
1 = Moderately Effective
2 = Effective

Found:

{unique_labels}
"""
    )


print("\nEffectiveness Label distribution:")


for label in [0, 1, 2]:

    count = (
        df[TARGET] == label
    ).sum()

    percentage = (
        count / len(df)
    ) * 100

    print(
        f"{label} = "
        f"{LABEL_NAMES[label]}: "
        f"{count} "
        f"({percentage:.2f}%)"
    )


# ============================================================
# 5. SHOW HOW EFFECTIVENESS LABEL RELATES TO DATA
# ============================================================

print_section(
    "5. EFFECTIVENESS LABEL INPUT VALUES"
)


print(
    """
IMPORTANT:

The Effectiveness Label already exists in the dataset.

The Random Forest does NOT manually calculate the label.

The six columns below are the INPUTS:

1. Farm Size
2. Average Yield
3. Crop Yield
4. Average Selling Price
5. Feedback Score
6. Subsidy Received

The Effectiveness Label is the TARGET that the model learns.
"""
)


label_summary = (

    df.groupby(TARGET)

    .agg({

        "Farm Size (ha)": [
            "mean",
            "min",
            "max"
        ],

        "Average Yield (tons/ha)": [
            "mean",
            "min",
            "max"
        ],

        "Crop Yield (tons)": [
            "mean",
            "min",
            "max"
        ],

        "Average Selling Price (₱/kg)": [
            "mean",
            "min",
            "max"
        ],

        "Feedback Score": [
            "mean",
            "min",
            "max"
        ],

        "Subsidy Received": [
            "mean",
            "min",
            "max"
        ]

    })

)


print(
    label_summary
)


# ============================================================
# 6. CREATE EXPLANATION / COMPUTATION TABLE
# ============================================================

print_section(
    "6. PREPARING EFFECTIVENESS COMPUTATION TABLE"
)


computation_df = df.copy()


# Human-readable label

computation_df["Effectiveness"] = (
    computation_df[TARGET]
    .map(LABEL_NAMES)
)


# ------------------------------------------------------------
# Yield comparison
# ------------------------------------------------------------
#
# Average Yield is tons/ha.
#
# Crop Yield is total tons.
#
# Therefore:
#
# Expected production =
# Farm Size × Average Yield
#
# Yield Difference =
# Crop Yield - Expected production
#
# Yield Change (%) =
# (Crop Yield - Expected production)
# / Expected production × 100
#
# ------------------------------------------------------------

computation_df[
    "Expected Production (tons)"
] = (

    computation_df["Farm Size (ha)"]
    *
    computation_df["Average Yield (tons/ha)"]

)


computation_df[
    "Yield Difference (tons)"
] = (

    computation_df["Crop Yield (tons)"]
    -
    computation_df["Expected Production (tons)"]

)


computation_df[
    "Yield Change (%)"
] = np.where(

    computation_df[
        "Expected Production (tons)"
    ] != 0,

    (

        computation_df["Yield Difference (tons)"]
        /
        computation_df[
            "Expected Production (tons)"
        ]

    ) * 100,

    0

)


# Round calculated values

computation_df[
    "Expected Production (tons)"
] = computation_df[
    "Expected Production (tons)"
].round(2)


computation_df[
    "Yield Difference (tons)"
] = computation_df[
    "Yield Difference (tons)"
].round(2)


computation_df[
    "Yield Change (%)"
] = computation_df[
    "Yield Change (%)"
].round(2)


# Put useful columns first

computation_columns = [

    "Farm Size (ha)",

    "Average Yield (tons/ha)",

    "Expected Production (tons)",

    "Crop Yield (tons)",

    "Yield Difference (tons)",

    "Yield Change (%)",

    "Average Selling Price (₱/kg)",

    "Feedback Score",

    "Subsidy Received",

    TARGET,

    "Effectiveness"

]


computation_df = computation_df[
    computation_columns
]


print(
    "\nSample effectiveness computation:"
)

print(
    computation_df
    .head(20)
    .to_string(index=False)
)


# ============================================================
# 7. ORIGINAL CLASS DISTRIBUTION
# ============================================================

print_section(
    "7. ORIGINAL CLASS DISTRIBUTION"
)


target_distribution = (

    df[TARGET]

    .value_counts()

    .sort_index()

)


target_distribution_display = pd.DataFrame({

    "Class": [

        LABEL_NAMES[value]

        for value in target_distribution.index

    ],

    "Encoded Value":
        target_distribution.index,

    "Count":
        target_distribution.values,

    "Percentage":
        (

            target_distribution.values
            /
            len(df)
            *
            100

        ).round(2)

})


print(

    target_distribution_display

    .to_string(index=False)

)


# ============================================================
# 8. PREPARE FEATURES
# ============================================================

print_section(
    "8. PREPARING FEATURES"
)


X = df[
    REQUIRED_FEATURES
].copy()


y = df[
    TARGET
].copy()


print("Features used by Random Forest:")


for feature in X.columns:

    print(
        f" - {feature}"
    )


print(
    f"\nTotal features: {X.shape[1]}"
)


# ============================================================
# 9. NUMERICAL VALIDATION
# ============================================================

print_section(
    "9. NUMERICAL FEATURE VALIDATION"
)


non_numeric_columns = (

    X

    .select_dtypes(
        exclude=[np.number]
    )

    .columns

    .tolist()

)


if non_numeric_columns:

    raise TypeError(

        "Non-numerical features detected:\n"

        +

        "\n".join(

            f" - {column}"

            for column
            in non_numeric_columns

        )

    )


print(
    "All model features are numerical."
)


# ============================================================
# 10. FEATURE INFORMATION
# ============================================================

print_section(
    "10. FEATURE INFORMATION"
)


for column in X.columns:

    print(

        f"{column}: "

        f"{X[column].nunique()} unique values | "

        f"min={X[column].min()} | "

        f"max={X[column].max()}"

    )


# ============================================================
# 11. SAVE FEATURE ORDER
# ============================================================

joblib.dump(

    X.columns.tolist(),

    FEATURES_FILE

)


print(
    f"\nFeature order saved to: "
    f"{FEATURES_FILE}"
)


# ============================================================
# 12. TRAIN / TEST SPLIT
# ============================================================

print_section(
    "11. TRAIN / TEST SPLIT"
)


X_train, X_test, y_train, y_test = (

    train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=RANDOM_STATE,

        stratify=y

    )

)


print(
    f"Training samples: {len(X_train)}"
)

print(
    f"Testing samples : {len(X_test)}"
)


print(
    "\nTraining distribution BEFORE SMOTE:"
)


train_distribution = (

    y_train

    .value_counts()

    .sort_index()

)


for label, count in train_distribution.items():

    print(

        f"{label} - "
        f"{LABEL_NAMES[label]}: "
        f"{count}"

    )


print(
    "\nTesting distribution:"
)


test_distribution = (

    y_test

    .value_counts()

    .sort_index()

)


for label, count in test_distribution.items():

    print(

        f"{label} - "
        f"{LABEL_NAMES[label]}: "
        f"{count}"

    )


# ============================================================
# 13. SMOTE CHECK
# ============================================================

print_section(
    "12. SMOTE BALANCING"
)


smote_check = SMOTE(
    random_state=RANDOM_STATE
)


X_train_smote_check, y_train_smote_check = (

    smote_check.fit_resample(

        X_train,

        y_train

    )

)


before_smote = (

    y_train

    .value_counts()

    .sort_index()

)


after_smote = (

    pd.Series(
        y_train_smote_check
    )

    .value_counts()

    .sort_index()

)


smote_distribution = pd.DataFrame({

    "Class": [

        LABEL_NAMES[label]

        for label in before_smote.index

    ],

    "Encoded Value":
        before_smote.index,

    "Before SMOTE":
        before_smote.values,

    "After SMOTE": [

        after_smote.get(
            label,
            0
        )

        for label
        in before_smote.index

    ]

})


print(
    smote_distribution
    .to_string(index=False)
)


# ============================================================
# 14. PIPELINE
# ============================================================

print_section(
    "13. BUILDING RANDOM FOREST PIPELINE"
)


pipeline = Pipeline([

    (

        "smote",

        SMOTE(
            random_state=RANDOM_STATE
        )

    ),

    (

        "classifier",

        RandomForestClassifier(

            random_state=RANDOM_STATE,

            n_jobs=-1

        )

    )

])


print(
    "Pipeline:"
)

print(
    "1. SMOTE"
)

print(
    "2. Random Forest Classifier"
)


# ============================================================
# 15. HYPERPARAMETER SEARCH
# ============================================================

print_section(
    "14. RANDOM FOREST HYPERPARAMETER SEARCH"
)


param_distributions = {

    "classifier__n_estimators": [

        100,
        200,
        300,
        400

    ],

    "classifier__max_depth": [

        5,
        8,
        10,
        12,
        15,
        None

    ],

    "classifier__min_samples_split": [

        2,
        5,
        10,
        15,
        20

    ],

    "classifier__min_samples_leaf": [

        1,
        2,
        4,
        6,
        8

    ],

    "classifier__max_features": [

        "sqrt",
        "log2"

    ],

    "classifier__bootstrap": [

        True

    ]

}


# ============================================================
# 16. CROSS VALIDATION
# ============================================================

cv_strategy = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=RANDOM_STATE

)


random_search = RandomizedSearchCV(

    estimator=pipeline,

    param_distributions=param_distributions,

    n_iter=40,

    scoring="f1_weighted",

    cv=cv_strategy,

    verbose=2,

    random_state=RANDOM_STATE,

    n_jobs=-1,

    return_train_score=True

)


# ============================================================
# 17. TRAIN MODEL
# ============================================================

print_section(
    "15. TRAINING RANDOM FOREST"
)


print(
    "Training Random Forest with SMOTE..."
)


random_search.fit(

    X_train,

    y_train

)


print(
    "\nTraining completed."
)


# ============================================================
# 18. BEST MODEL
# ============================================================

print_section(
    "16. BEST MODEL"
)


best_model = (

    random_search
    .best_estimator_

)


print(
    "Best parameters:"
)


for parameter, value in (

    random_search
    .best_params_
    .items()

):

    print(
        f"{parameter}: {value}"
    )


print(

    f"\nBest CV F1 Score: "
    f"{random_search.best_score_:.4f}"

)


# ============================================================
# 19. TEST PREDICTION
# ============================================================

print_section(
    "17. TEST SET PREDICTIONS"
)


y_pred = best_model.predict(
    X_test
)


print(
    "Predictions generated successfully."
)


# ============================================================
# 20. CLASSIFICATION METRICS
# ============================================================

print_section(
    "18. CLASSIFICATION METRICS"
)


accuracy = accuracy_score(
    y_test,
    y_pred
)


precision = precision_score(

    y_test,

    y_pred,

    average="weighted",

    zero_division=0

)


recall = recall_score(

    y_test,

    y_pred,

    average="weighted",

    zero_division=0

)


f1 = f1_score(

    y_test,

    y_pred,

    average="weighted",

    zero_division=0

)


print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)


# ============================================================
# 21. CLASSIFICATION REPORT
# ============================================================

print_section(
    "19. CLASSIFICATION REPORT"
)


classification_report_text = (

    classification_report(

        y_test,

        y_pred,

        labels=[0, 1, 2],

        target_names=[

            LABEL_NAMES[0],

            LABEL_NAMES[1],

            LABEL_NAMES[2]

        ],

        zero_division=0

    )

)


print(
    classification_report_text
)


# ============================================================
# 22. CONFUSION MATRIX
# ============================================================

print_section(
    "20. CONFUSION MATRIX"
)


cm = confusion_matrix(

    y_test,

    y_pred,

    labels=[0, 1, 2]

)


print(cm)


fig, ax = plt.subplots(

    figsize=(8, 6)

)


display = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=[

        LABEL_NAMES[0],

        LABEL_NAMES[1],

        LABEL_NAMES[2]

    ]

)


display.plot(

    ax=ax,

    values_format="d"

)


ax.set_title(
    "Random Forest Confusion Matrix"
)


plt.tight_layout()


plt.savefig(

    CONFUSION_MATRIX_FILE,

    dpi=300,

    bbox_inches="tight"

)


plt.close()


print(

    f"Confusion matrix saved to: "
    f"{CONFUSION_MATRIX_FILE}"

)


# ============================================================
# 23. ERROR METRICS
# ============================================================

print_section(
    "21. ERROR METRICS"
)


mae = mean_absolute_error(

    y_test,

    y_pred

)


mse = mean_squared_error(

    y_test,

    y_pred

)


rmse = np.sqrt(mse)


r2 = r2_score(

    y_test,

    y_pred

)


print(
    f"MAE : {mae:.4f}"
)

print(
    f"MSE : {mse:.4f}"
)

print(
    f"RMSE: {rmse:.4f}"
)

print(
    f"R²  : {r2:.4f}"
)


# ============================================================
# 24. ACTUAL VS PREDICTED
# ============================================================

print_section(
    "22. ACTUAL VS PREDICTED"
)


results_df = pd.DataFrame({

    "Actual":
        y_test.values,

    "Predicted":
        y_pred

})


results_df["Error"] = (

    results_df["Predicted"]

    -

    results_df["Actual"]

)


results_df["Absolute Error"] = (

    results_df["Error"]

    .abs()

)


results_df["Actual Label"] = (

    results_df["Actual"]

    .map(LABEL_NAMES)

)


results_df["Predicted Label"] = (

    results_df["Predicted"]

    .map(LABEL_NAMES)

)


print(

    results_df
    .head(20)
    .to_string(index=False)

)


# ============================================================
# 25. CROSS VALIDATION
# ============================================================

print_section(
    "23. CROSS-VALIDATION"
)


cv_scores = cross_val_score(

    best_model,

    X_train,

    y_train,

    cv=cv_strategy,

    scoring="f1_weighted",

    n_jobs=-1

)


print(
    "Cross-validation F1 scores:"
)


for index, score in enumerate(

    cv_scores,

    start=1

):

    print(

        f"Fold {index}: "
        f"{score:.4f}"

    )


cv_mean = cv_scores.mean()

cv_std = cv_scores.std()


print(

    f"\nMean CV F1 Score: "
    f"{cv_mean:.4f}"

)


print(

    f"CV Standard Deviation: "
    f"{cv_std:.4f}"

)


# ============================================================
# 26. LEARNING CURVE
# ============================================================

print_section(
    "24. LEARNING CURVE"
)


train_sizes, train_scores, validation_scores = (

    learning_curve(

        best_model,

        X_train,

        y_train,

        cv=cv_strategy,

        scoring="f1_weighted",

        train_sizes=np.linspace(

            0.1,

            1.0,

            5

        ),

        n_jobs=-1

    )

)


train_mean = train_scores.mean(
    axis=1
)


train_std = train_scores.std(
    axis=1
)


validation_mean = validation_scores.mean(
    axis=1
)


validation_std = validation_scores.std(
    axis=1
)


plt.figure(

    figsize=(9, 6)

)


plt.plot(

    train_sizes,

    train_mean,

    marker="o",

    label="Training F1"

)


plt.plot(

    train_sizes,

    validation_mean,

    marker="o",

    label="Validation F1"

)


plt.fill_between(

    train_sizes,

    train_mean - train_std,

    train_mean + train_std,

    alpha=0.15

)


plt.fill_between(

    train_sizes,

    validation_mean - validation_std,

    validation_mean + validation_std,

    alpha=0.15

)


plt.xlabel(
    "Number of Training Samples"
)


plt.ylabel(
    "Weighted F1 Score"
)


plt.title(
    "Random Forest Learning Curve"
)


plt.legend()


plt.grid(

    True,

    alpha=0.3

)


plt.tight_layout()


plt.savefig(

    LEARNING_CURVE_FILE,

    dpi=300,

    bbox_inches="tight"

)


plt.close()


print(

    f"Learning curve saved to: "
    f"{LEARNING_CURVE_FILE}"

)


# ============================================================
# 27. FEATURE IMPORTANCE
# ============================================================

print_section(
    "25. FEATURE IMPORTANCE"
)


rf_model = (

    best_model
    .named_steps["classifier"]

)


feature_importance = (

    rf_model
    .feature_importances_

)


feature_importance_df = pd.DataFrame({

    "Feature":
        X.columns,

    "Importance":
        feature_importance

})


feature_importance_df = (

    feature_importance_df

    .sort_values(

        by="Importance",

        ascending=False

    )

    .reset_index(drop=True)

)


print(

    feature_importance_df
    .to_string(index=False)

)


feature_importance_df.to_excel(

    FEATURE_IMPORTANCE_FILE,

    index=False

)


print(

    f"\nFeature importance saved to: "
    f"{FEATURE_IMPORTANCE_FILE}"

)


# ============================================================
# 28. METRICS TABLE
# ============================================================

metrics_df = pd.DataFrame({

    "Metric": [

        "Accuracy",

        "Precision",

        "Recall",

        "F1 Score",

        "MAE",

        "MSE",

        "RMSE",

        "R²",

        "Mean CV F1",

        "CV Standard Deviation"

    ],

    "Value": [

        accuracy,

        precision,

        recall,

        f1,

        mae,

        mse,

        rmse,

        r2,

        cv_mean,

        cv_std

    ]

})


# ============================================================
# 29. EXPORT EVERYTHING
# ============================================================

print_section(
    "26. EXPORTING TRAINING RESULTS"
)


with pd.ExcelWriter(

    TRAINING_METRICS_FILE,

    engine="openpyxl"

) as writer:


    # Metrics

    metrics_df.to_excel(

        writer,

        sheet_name="Metrics",

        index=False

    )


    # Actual vs predicted

    results_df.to_excel(

        writer,

        sheet_name="Actual vs Predicted",

        index=False

    )


    # Original class balance

    target_distribution_display.to_excel(

        writer,

        sheet_name="Original Class Balance",

        index=False

    )


    # SMOTE

    smote_distribution.to_excel(

        writer,

        sheet_name="SMOTE Balance",

        index=False

    )


    # Feature importance

    feature_importance_df.to_excel(

        writer,

        sheet_name="Feature Importance",

        index=False

    )


    # NEW:
    # Detailed effectiveness computation

    computation_df.to_excel(

        writer,

        sheet_name="Effectiveness Computation",

        index=False

    )


print(

    f"Training results saved to: "
    f"{TRAINING_METRICS_FILE}"

)


# ============================================================
# 30. SAVE MODEL
# ============================================================

print_section(
    "27. SAVING MODEL"
)


joblib.dump(

    best_model,

    MODEL_FILE

)


print(
    "Model saved successfully."
)


print(
    os.path.abspath(
        MODEL_FILE
    )
)


# ============================================================
# 31. FINAL SUMMARY
# ============================================================

print_section(
    "28. FINAL MODEL SUMMARY"
)


print(
    "AgriSubsidy Effectiveness Prediction"
)


print(
    "Model: Random Forest Classifier"
)


print(
    "Balancing: SMOTE"
)


print(
    f"Dataset Rows: {len(df)}"
)


print(
    f"Training Samples: {len(X_train)}"
)


print(
    f"Testing Samples: {len(X_test)}"
)


print(
    f"Features Used: {X.shape[1]}"
)


print(
    f"Accuracy: {accuracy:.4f}"
)


print(
    f"Precision: {precision:.4f}"
)


print(
    f"Recall: {recall:.4f}"
)


print(
    f"F1 Score: {f1:.4f}"
)


print(
    f"MAE: {mae:.4f}"
)


print(
    f"MSE: {mse:.4f}"
)


print(
    f"RMSE: {rmse:.4f}"
)


print(
    f"R²: {r2:.4f}"
)


print(
    f"Mean CV F1: {cv_mean:.4f}"
)


print(
    f"CV Std: {cv_std:.4f}"
)


print("\nFiles generated:")


print(
    f" - {MODEL_FILE}"
)


print(
    f" - {FEATURES_FILE}"
)


print(
    f" - {CONFUSION_MATRIX_FILE}"
)


print(
    f" - {LEARNING_CURVE_FILE}"
)


print(
    f" - {FEATURE_IMPORTANCE_FILE}"
)


print(
    f" - {TRAINING_METRICS_FILE}"
)


print_section(
    "TRAINING COMPLETE"
)