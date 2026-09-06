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
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.preprocessing import OneHotEncoder

# IMPORTANT:
# Use imblearn Pipeline instead of sklearn Pipeline
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


# ============================================================
# RANDOM SEEDS
# ============================================================

random.seed(42)
np.random.seed(42)


# ============================================================
# FILE SETTINGS
# ============================================================

TRAIN_FILE = "datasets/subsidy_dataset2_initial.xlsx"
VALIDATION_FILE = "datasets/subsidy_validation_datasets_initial.xlsx"

TARGET = "Effectiveness Label"


# ============================================================
# LOAD DATASETS
# ============================================================

train_df = pd.read_excel(TRAIN_FILE)
validation_df = pd.read_excel(VALIDATION_FILE)


print("\n========================================")
print("TRAINING DATASET")
print("========================================")

print(train_df.head())
print("Shape:", train_df.shape)


print("\n========================================")
print("VALIDATION DATASET")
print("========================================")

print(validation_df.head())
print("Shape:", validation_df.shape)


# ============================================================
# CHECK TARGET DISTRIBUTION
# ============================================================

print("\n========================================")
print("TRAINING CLASS DISTRIBUTION")
print("========================================")

print(train_df[TARGET].value_counts())

print("\nTraining Class Distribution (%)")

print(
    train_df[TARGET]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


print("\n========================================")
print("VALIDATION CLASS DISTRIBUTION")
print("========================================")

print(validation_df[TARGET].value_counts())

print("\nValidation Class Distribution (%)")

print(
    validation_df[TARGET]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ============================================================
# FEATURES AND TARGET
# ============================================================

X = train_df.drop(columns=[TARGET])
y = train_df[TARGET]

X_validation = validation_df.drop(columns=[TARGET])
y_validation = validation_df[TARGET]


# ============================================================
# FEATURE TYPES
# ============================================================

categorical_cols = [
    "Subsidy Type",
    "Pest",
    "Calamity",
    "Subsidy Received",
]

numerical_cols = [
    "Farm Size (ha)",
    "Average Yield (tons/ha)",
    "Crop Yield (bags/ha)",
    "Average Selling Price (₱/kg)",
    "Feedback Score"
]


# ============================================================
# PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
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


print("\n========================================")
print("TRAIN / TEST SPLIT")
print("========================================")

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


print("\nTraining class distribution:")

print(y_train.value_counts())


print("\nTesting class distribution:")

print(y_test.value_counts())


# ============================================================
# PIPELINE
#
# ORIGINAL DATA
#       ↓
# PREPROCESSING
#       ↓
# SMOTE
#       ↓
# RANDOM FOREST
# ============================================================

pipeline = Pipeline([
    
    (
        "preprocessor",
        preprocessor
    ),

    (
        "smote",
        SMOTE(
            random_state=42
        )
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
# RANDOM FOREST PARAMETERS
# ============================================================

params = {

    "classifier__n_estimators": [
        100,
        200,
        300,
        500
    ],

    "classifier__max_depth": [
        None,
        10,
        15,
        20,
        30
    ],

    "classifier__min_samples_split": [
        2,
        5,
        10
    ],

    "classifier__min_samples_leaf": [
        1,
        2,
        4
    ],

    "classifier__max_features": [
        "sqrt",
        "log2"
    ],

    "classifier__bootstrap": [
        True,
        False
    ]
}


# ============================================================
# RANDOMIZED SEARCH
# ============================================================

print("\n========================================")
print("RANDOMIZED SEARCH")
print("========================================")

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

print("\nTraining model with SMOTE...")

search.fit(
    X_train,
    y_train
)


print("\n========================================")
print("BEST PARAMETERS")
print("========================================")

print(search.best_params_)

print("\nBest Cross-Validation Score:")
print(f"{search.best_score_ * 100:.2f}%")

model = search.best_estimator_


# ============================================================
# TRAINING ACCURACY
# ============================================================

train_acc = model.score(
    X_train,
    y_train
)

test_acc = model.score(
    X_test,
    y_test
)

print("\n========================================")
print("TRAIN / TEST ACCURACY")
print("========================================")

print(
    f"Train Accuracy : {train_acc * 100:.2f}%"
)

print(
    f"Test Accuracy  : {test_acc * 100:.2f}%"
)


# ============================================================
# TEST PREDICTION
# ============================================================

y_pred = model.predict(
    X_test
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n========================================")
print("TEST CLASSIFICATION REPORT")
print("========================================")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ============================================================
# TEST CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)


print("\n========================================")
print("TEST CONFUSION MATRIX")
print("========================================")

print(cm)


plt.figure(figsize=(7, 6))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

disp.plot(
    cmap="Blues",
    values_format="d"
)

plt.title(
    "Random Forest - Test Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "confusion_matrix_test.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# CROSS VALIDATION
# ============================================================

print("\n========================================")
print("CROSS VALIDATION")
print("========================================")

cv = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)


print("CV Scores:")

print(cv)


print(
    f"\nCV Mean Accuracy: {cv.mean() * 100:.2f}%"
)


print(
    f"CV Standard Deviation: {cv.std() * 100:.2f}%"
)


# ============================================================
# LEARNING CURVE
# ============================================================

print("\n========================================")
print("LEARNING CURVE")
print("========================================")

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

train_std = train_scores.std(
    axis=1
)

valid_mean = valid_scores.mean(
    axis=1
)

valid_std = valid_scores.std(
    axis=1
)


plt.figure(figsize=(8, 5))


plt.plot(
    sizes,
    train_mean,
    marker="o",
    label="Training Accuracy"
)


plt.plot(
    sizes,
    valid_mean,
    marker="o",
    label="Validation Accuracy"
)


plt.fill_between(
    sizes,
    train_mean - train_std,
    train_mean + train_std,
    alpha=0.15
)


plt.fill_between(
    sizes,
    valid_mean - valid_std,
    valid_mean + valid_std,
    alpha=0.15
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
    True,
    alpha=0.3
)

plt.legend()


plt.tight_layout()


plt.savefig(
    "learning_curve.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# EXTERNAL VALIDATION
# ============================================================

print("\n========================================")
print("EXTERNAL VALIDATION")
print("========================================")


val_pred = model.predict(
    X_validation
)


val_acc = accuracy_score(
    y_validation,
    val_pred
)


print(
    f"External Validation Accuracy : "
    f"{val_acc * 100:.2f}%"
)


# ============================================================
# EXTERNAL VALIDATION REPORT
# ============================================================

print("\n========================================")
print("EXTERNAL VALIDATION CLASSIFICATION REPORT")
print("========================================")


print(
    classification_report(
        y_validation,
        val_pred
    )
)


# ============================================================
# EXTERNAL VALIDATION CONFUSION MATRIX
# ============================================================

val_cm = confusion_matrix(
    y_validation,
    val_pred
)


print("\n========================================")
print("VALIDATION CONFUSION MATRIX")
print("========================================")

print(val_cm)


plt.figure(figsize=(7, 6))


val_disp = ConfusionMatrixDisplay(
    confusion_matrix=val_cm,
    display_labels=model.classes_
)


val_disp.plot(
    cmap="Blues",
    values_format="d"
)


plt.title(
    "Random Forest - External Validation Confusion Matrix"
)


plt.tight_layout()


plt.savefig(
    "confusion_matrix_validation.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n========================================")
print("FEATURE IMPORTANCE")
print("========================================")


feature_names = (
    model
    .named_steps["preprocessor"]
    .get_feature_names_out()
)


feature_importance = (
    model
    .named_steps["classifier"]
    .feature_importances_
)


print(
    "Number of Features:",
    len(feature_names)
)


print(
    "Number of Importances:",
    len(feature_importance)
)


fi = pd.DataFrame({

    "Feature": feature_names,

    "Importance": feature_importance

})


fi = fi.sort_values(
    by="Importance",
    ascending=False
)


# ============================================================
# SAVE FEATURE IMPORTANCE
# ============================================================

fi.to_excel(
    "feature_importance.xlsx",
    index=False
)


print("\nTop 10 Most Important Features:")

print(
    fi.head(10)
)


# ============================================================
# FEATURE IMPORTANCE GRAPH
# ============================================================

top_features = fi.head(10).sort_values(
    by="Importance"
)


plt.figure(figsize=(9, 6))


plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)


plt.xlabel(
    "Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Top 10 Feature Importance - Random Forest"
)


plt.tight_layout()


plt.savefig(
    "feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# METRICS EXPORT
# ============================================================

metrics = pd.DataFrame({

    "Metric": [
        "Train Accuracy",
        "Test Accuracy",
        "External Validation Accuracy",
        "Cross Validation Mean",
        "Cross Validation Std"
    ],

    "Value": [
        train_acc,
        test_acc,
        val_acc,
        cv.mean(),
        cv.std()
    ]

})


# ============================================================
# SAVE METRICS
# ============================================================

metrics.to_excel(
    "training_metrics.xlsx",
    index=False
)


# ============================================================
# SAVE BEST MODEL
# ============================================================

joblib.dump(
    model,
    "random_forest_subsidy_smote.pkl"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n========================================")
print("TRAINING COMPLETE")
print("========================================")


print(
    f"Train Accuracy              : "
    f"{train_acc * 100:.2f}%"
)


print(
    f"Test Accuracy               : "
    f"{test_acc * 100:.2f}%"
)


print(
    f"External Validation Accuracy: "
    f"{val_acc * 100:.2f}%"
)


print(
    f"Cross Validation Mean       : "
    f"{cv.mean() * 100:.2f}%"
)


print(
    f"Cross Validation Std        : "
    f"{cv.std() * 100:.2f}%"
)


print("\nGenerated Files:")

print(
    "- random_forest_subsidy_smote.pkl"
)

print(
    "- confusion_matrix_test.png"
)

print(
    "- confusion_matrix_validation.png"
)

print(
    "- learning_curve.png"
)

print(
    "- feature_importance.png"
)

print(
    "- feature_importance.xlsx"
)

print(
    "- training_metrics.xlsx"
)

print("\n========================================")
print("DONE")
print("========================================")