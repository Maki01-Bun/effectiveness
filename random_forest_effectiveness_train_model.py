import random
import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    cross_val_score,
    learning_curve,
    train_test_split
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# RANDOM SEEDS
# ============================================================

random.seed(42)
np.random.seed(42)


# ============================================================
# FILE AND TARGET SETTINGS
# ============================================================

TRAIN_FILE = "datasets/subsidy_dataset2_initial.xlsx"
TARGET = "Effectiveness Label"


# ============================================================
# EFFECTIVENESS LABELS
# ============================================================

LABEL_NAMES = {
    0: "Not Effective",
    1: "Moderately Effective",
    2: "Effective"
}


# ============================================================
# LOAD DATASET
# ============================================================

train_df = pd.read_excel(TRAIN_FILE)

print("\n==============================")
print("TRAINING DATASET")
print("==============================")

print(train_df.head())
print("\nDataset Shape:", train_df.shape)


# ============================================================
# DATASET INFORMATION
# ============================================================

print("\n==============================")
print("DATASET INFORMATION")
print("==============================")

print(train_df.info())

print("\nMissing Values:")
print(train_df.isnull().sum())


# ============================================================
# CHECK AND NORMALIZE EFFECTIVENESS LABELS
# ============================================================

print("\n==============================")
print("EFFECTIVENESS LABELS")
print("==============================")

print("0 = Not Effective")
print("1 = Moderately Effective")
print("2 = Effective")

print("\nOriginal Target Values:")
print(train_df[TARGET].value_counts(dropna=False))


# ------------------------------------------------------------
# CONVERT TARGET TO NUMERIC
# ------------------------------------------------------------

def normalize_effectiveness_label(value):

    if pd.isna(value):
        return np.nan

    # Already numeric
    if isinstance(value, (int, np.integer)):
        value = int(value)

        if value in [0, 1, 2]:
            return value

    if isinstance(value, (float, np.floating)):
        if value in [0.0, 1.0, 2.0]:
            return int(value)

    # String values
    value = str(value).strip().lower()

    label_mapping = {

        "0": 0,
        "not effective": 0,

        "1": 1,
        "moderately effective": 1,

        "2": 2,
        "effective": 2
    }

    return label_mapping.get(value, np.nan)


train_df[TARGET] = (
    train_df[TARGET]
    .apply(normalize_effectiveness_label)
)


# ------------------------------------------------------------
# CHECK FOR INVALID LABELS
# ------------------------------------------------------------

invalid_labels = train_df[TARGET].isna().sum()

if invalid_labels > 0:

    print(
        f"\nWARNING: {invalid_labels} invalid/missing "
        f"effectiveness labels found."
    )

    print(
        "\nRows with invalid labels:"
    )

    print(
        train_df[
            train_df[TARGET].isna()
        ]
    )

    # Remove invalid target rows
    train_df = train_df.dropna(
        subset=[TARGET]
    )


# Convert to integer
train_df[TARGET] = (
    train_df[TARGET]
    .astype(int)
)


# ------------------------------------------------------------
# TARGET DISTRIBUTION
# ------------------------------------------------------------

print("\nNormalized Target Distribution:")

print(
    train_df[TARGET]
    .value_counts()
    .sort_index()
)


# ------------------------------------------------------------
# VERIFY ALL THREE CLASSES
# ------------------------------------------------------------

required_classes = {0, 1, 2}

actual_classes = set(
    train_df[TARGET].unique()
)

missing_classes = (
    required_classes - actual_classes
)

if missing_classes:

    raise ValueError(
        f"\nERROR: Missing effectiveness classes: "
        f"{sorted(missing_classes)}\n"
        f"The dataset must contain all three classes:\n"
        f"0 = Not Effective\n"
        f"1 = Moderately Effective\n"
        f"2 = Effective"
    )


# ============================================================
# FEATURES AND TARGET
# ============================================================

X = train_df.drop(
    columns=[TARGET]
)

y = train_df[TARGET]


# ============================================================
# COLUMN TYPES
# ============================================================

categorical_cols = [
    "Subsidy Received"
]

numerical_cols = [
    "Farm Size (ha)",
    "Average Yield (bags/ha)",
    "Crop Yield (bags/ha)",
    "Average Selling Price (₱/kg)",
    "Feedback Score"
]


# ============================================================
# PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_cols
        ),

        (
            "num",
            "passthrough",
            numerical_cols
        )
    ]
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)


print("\n==============================")
print("TRAIN / TEST SPLIT")
print("==============================")

print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# ============================================================
# RANDOM FOREST CLASSIFIER
# ANTI-OVERFITTING SETTINGS
# ============================================================

pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),

    (
        "classifier",
        RandomForestClassifier(
            random_state=42,
            n_jobs=-1
        )
    )
])


# ============================================================
# HYPERPARAMETER SEARCH
# ============================================================

params = {

    # Number of trees
    "classifier__n_estimators": [
        100,
        200,
        300
    ],

    # Limit tree complexity
    "classifier__max_depth": [
        5,
        8,
        10,
        12,
        15
    ],

    # Require more samples before splitting
    "classifier__min_samples_split": [
        5,
        10,
        15,
        20
    ],

    # Require more samples in leaf nodes
    "classifier__min_samples_leaf": [
        2,
        4,
        6,
        8
    ],

    # Number of features considered at each split
    "classifier__max_features": [
        "sqrt",
        "log2"
    ],

    # Bootstrap sampling
    "classifier__bootstrap": [
        True
    ]
}


# ============================================================
# RANDOMIZED SEARCH
# ============================================================

print("\n==============================")
print("HYPERPARAMETER TUNING")
print("==============================")

search = RandomizedSearchCV(

    estimator=pipeline,

    param_distributions=params,

    n_iter=40,

    cv=5,

    scoring="accuracy",

    random_state=42,

    n_jobs=-1,

    verbose=1
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\n==============================")
print("TRAINING RANDOM FOREST")
print("==============================")

search.fit(
    X_train,
    y_train
)


# Best model
model = search.best_estimator_


# ============================================================
# BEST PARAMETERS
# ============================================================

print("\n==============================")
print("BEST PARAMETERS")
print("==============================")

print(
    search.best_params_
)


# ============================================================
# CLASSIFICATION EVALUATION
# ============================================================

print("\n==============================")
print("CLASSIFICATION EVALUATION")
print("==============================")


# Training predictions
y_train_pred = model.predict(
    X_train
)


# Testing predictions
y_pred = model.predict(
    X_test
)


# Training accuracy
train_acc = accuracy_score(
    y_train,
    y_train_pred
)


# Testing accuracy
test_acc = accuracy_score(
    y_test,
    y_pred
)


print(
    f"Train Accuracy : {train_acc * 100:.2f}%"
)

print(
    f"Test Accuracy  : {test_acc * 100:.2f}%"
)


# ============================================================
# OVERFITTING GAP
# ============================================================

overfit_gap = (
    train_acc - test_acc
)


print(
    f"Train-Test Gap : {overfit_gap * 100:.2f} percentage points"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")


print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        target_names=[
            "Not Effective",
            "Moderately Effective",
            "Effective"
        ],
        digits=4
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")


cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1, 2]
)


print(cm)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Not Effective",
        "Moderately Effective",
        "Effective"
    ]
)


disp.plot()


plt.title(
    "Random Forest - Effectiveness Label"
)


plt.savefig(
    "confusion_matrix_test.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# PRECISION, RECALL, F1
# ============================================================

report = classification_report(
    y_test,
    y_pred,
    labels=[0, 1, 2],
    target_names=[
        "Not Effective",
        "Moderately Effective",
        "Effective"
    ],
    output_dict=True
)


print("\n==============================")
print("CLASSIFICATION METRICS")
print("==============================")


print(
    f"Accuracy : {report['accuracy']:.4f}"
)

print(
    f"Precision: {report['weighted avg']['precision']:.4f}"
)

print(
    f"Recall   : {report['weighted avg']['recall']:.4f}"
)

print(
    f"F1-Score : {report['weighted avg']['f1-score']:.4f}"
)


# ============================================================
# SPECIFICITY
# ============================================================

print("\n==============================")
print("SPECIFICITY")
print("==============================")


specificity_values = []


for i in range(len(cm)):

    true_negative = (
        cm.sum()
        - cm[i, :].sum()
        - cm[:, i].sum()
        + cm[i, i]
    )

    false_positive = (
        cm[:, i].sum()
        - cm[i, i]
    )

    specificity = (
        true_negative
        / (true_negative + false_positive)
        if (true_negative + false_positive) > 0
        else 0
    )

    specificity_values.append(
        specificity
    )

    print(
        f"{LABEL_NAMES[i]} Specificity: "
        f"{specificity:.4f}"
    )


weighted_specificity = np.mean(
    specificity_values
)


print(
    f"\nAverage Specificity: "
    f"{weighted_specificity:.4f}"
)


# ============================================================
# REGRESSION-STYLE EVALUATION
# ============================================================

print("\n==============================")
print("REGRESSION-STYLE EVALUATION")
print("==============================")


print("\nEffectiveness Label Encoding:")

print("0 = Not Effective")
print("1 = Moderately Effective")
print("2 = Effective")


# ------------------------------------------------------------
# MAE
# ------------------------------------------------------------

mae = mean_absolute_error(
    y_test,
    y_pred
)


# ------------------------------------------------------------
# MSE
# ------------------------------------------------------------

mse = mean_squared_error(
    y_test,
    y_pred
)


# ------------------------------------------------------------
# RMSE
# ------------------------------------------------------------

rmse = np.sqrt(
    mse
)


# ------------------------------------------------------------
# R2
# ------------------------------------------------------------

r2 = r2_score(
    y_test,
    y_pred
)


print("\nRegression Metrics:")

print(
    f"MAE  : {mae:.4f}"
)

print(
    f"MSE  : {mse:.4f}"
)

print(
    f"RMSE : {rmse:.4f}"
)

print(
    f"R²   : {r2:.4f}"
)


# ============================================================
# REGRESSION INTERPRETATION
# ============================================================

print("\n==============================")
print("REGRESSION INTERPRETATION")
print("==============================")


print(
    f"\nMAE Interpretation:"
)

print(
    f"The predictions differ from the actual "
    f"effectiveness labels by an average of "
    f"{mae:.4f} label levels."
)


print(
    f"\nMSE Interpretation:"
)

print(
    f"The average squared prediction error is "
    f"{mse:.4f}."
)


print(
    f"\nRMSE Interpretation:"
)

print(
    f"The typical prediction error is "
    f"{rmse:.4f} effectiveness-label levels."
)


print(
    f"\nR² Interpretation:"
)

print(
    f"The model explains approximately "
    f"{r2 * 100:.2f}% of the variation "
    f"in the numerical effectiveness labels."
)


# ============================================================
# CROSS VALIDATION
# ============================================================

print("\n==============================")
print("5-FOLD CROSS VALIDATION")
print("==============================")


cv = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)


print(
    "CV Scores:"
)

print(cv)


cv_mean = cv.mean()

cv_std = cv.std()


print(
    f"\nCV Mean Accuracy: "
    f"{cv_mean * 100:.2f}%"
)


print(
    f"CV Std. Deviation: "
    f"{cv_std * 100:.2f}%"
)


# ============================================================
# LEARNING CURVE
# ============================================================

print("\n==============================")
print("LEARNING CURVE")
print("==============================")


sizes, train_scores, valid_scores = learning_curve(

    model,

    X,

    y,

    cv=5,

    train_sizes=np.linspace(
        0.1,
        1.0,
        10
    ),

    scoring="accuracy",

    n_jobs=-1
)


train_mean = train_scores.mean(
    axis=1
)

valid_mean = valid_scores.mean(
    axis=1
)


plt.figure(
    figsize=(8, 5)
)


plt.plot(
    sizes,
    train_mean,
    marker="o",
    label="Training"
)


plt.plot(
    sizes,
    valid_mean,
    marker="o",
    label="Validation"
)


plt.xlabel(
    "Training Samples"
)

plt.ylabel(
    "Accuracy"
)


plt.title(
    "Random Forest Learning Curve"
)


plt.grid(
    True
)


plt.legend()


plt.savefig(
    "learning_curve.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n==============================")
print("FEATURE IMPORTANCE")
print("==============================")


feature_names = (
    model
    .named_steps[
        "preprocessor"
    ]
    .get_feature_names_out()
)


feature_importance = (
    model
    .named_steps[
        "classifier"
    ]
    .feature_importances_
)


fi = pd.DataFrame({

    "Feature":
        feature_names,

    "Importance":
        feature_importance
})


fi = fi.sort_values(
    by="Importance",
    ascending=False
)


print("\nTop 10 Most Important Features:")

print(
    fi.head(10)
)


# Save feature importance
fi.to_excel(
    "feature_importance.xlsx",
    index=False
)


# ============================================================
# METRICS EXPORT
# ============================================================

print("\n==============================")
print("EXPORTING METRICS")
print("==============================")


metrics = pd.DataFrame({

    "Metric": [

        "Train Accuracy",

        "Test Accuracy",

        "Train-Test Gap",

        "Cross Validation Accuracy",

        "Precision",

        "Recall",

        "F1-Score",

        "Specificity",

        "MAE",

        "MSE",

        "RMSE",

        "R2"

    ],

    "Value": [

        train_acc,

        test_acc,

        overfit_gap,

        cv_mean,

        report[
            "weighted avg"
        ][
            "precision"
        ],

        report[
            "weighted avg"
        ][
            "recall"
        ],

        report[
            "weighted avg"
        ][
            "f1-score"
        ],

        weighted_specificity,

        mae,

        mse,

        rmse,

        r2

    ]

})


metrics.to_excel(
    "training_metrics.xlsx",
    index=False
)


print(
    "Metrics saved to:"
)

print(
    "training_metrics.xlsx"
)


# ============================================================
# SAVE MODEL
# ============================================================

print("\n==============================")
print("SAVING MODEL")
print("==============================")


joblib.dump(
    model,
    "random_forest_subsidy.pkl"
)


print(
    "Model saved as:"
)

print(
    "random_forest_subsidy.pkl"
)


# ============================================================
# GENERATED FILES
# ============================================================

print("\n==============================")
print("TRAINING COMPLETE!")
print("==============================")


print("\nGenerated Files:")

print(
    "- random_forest_subsidy.pkl"
)

print(
    "- confusion_matrix_test.png"
)

print(
    "- learning_curve.png"
)

print(
    "- feature_importance.xlsx"
)

print(
    "- training_metrics.xlsx"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n==============================")
print("FINAL MODEL SUMMARY")
print("==============================")


print(
    f"Training Accuracy : "
    f"{train_acc * 100:.2f}%"
)

print(
    f"Testing Accuracy  : "
    f"{test_acc * 100:.2f}%"
)

print(
    f"CV Accuracy       : "
    f"{cv_mean * 100:.2f}%"
)

print(
    f"Specificity       : "
    f"{weighted_specificity:.4f}"
)

print(
    f"MAE               : "
    f"{mae:.4f}"
)

print(
    f"MSE               : "
    f"{mse:.4f}"
)

print(
    f"RMSE              : "
    f"{rmse:.4f}"
)

print(
    f"R²                : "
    f"{r2:.4f}"
)