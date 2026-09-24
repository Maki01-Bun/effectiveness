import os
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
    cross_val_score
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.inspection import permutation_importance

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


warnings.filterwarnings("ignore")


# ============================================================
# 2. CONFIGURATION
# ============================================================

TRAIN_FILE = "datasets/subsidy_dataset_ml_ready.xlsx"

SHEET_NAME = "Sheet1"

TARGET = "Effectiveness Label"

OUTPUT_DIR = "outputs"

RANDOM_STATE = 42


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 3. CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "Not Effective",
    "Moderately Effective",
    "Effective"
]


CLASS_LABELS = [0, 1, 2]


# ============================================================
# 4. LOAD DATASET
# ============================================================

print("\n============================================================")
print("LOADING DATASET")
print("============================================================")


df = pd.read_excel(
    TRAIN_FILE,
    sheet_name=SHEET_NAME
)


print(
    f"Dataset shape: {df.shape}"
)


print("\nDataset columns:")

for col in df.columns:
    print(" -", col)


# ============================================================
# 5. BASE FEATURES
# ============================================================

BASE_FEATURES = [

    "Farm Size (ha)",

    "Average Yield (tons/ha)",

    "Crop Yield (tons)",

    "Average Selling Price (₱/kg)",

    "Subsidy Received",

    "Q1",
    "Q2",
    "Q3",
    "Q4",
    "Q5",
    "Q6",
    "Q7",
    "Q8",
    "Q9",
    "Q10"
]


# ============================================================
# 6. VERIFY REQUIRED COLUMNS
# ============================================================

required_columns = BASE_FEATURES + [TARGET]


missing_columns = [

    col

    for col in required_columns

    if col not in df.columns

]


if missing_columns:

    raise ValueError(

        "Missing required columns:\n"

        + "\n".join(missing_columns)

    )


# ============================================================
# 7. CLEAN DATA
# ============================================================

print("\n============================================================")
print("DATA CLEANING")
print("============================================================")


data = df[
    required_columns
].copy()


for col in required_columns:

    data[col] = pd.to_numeric(

        data[col],

        errors="coerce"

    )


before_rows = len(data)


data = data.dropna()


after_rows = len(data)


print(
    f"Rows before cleaning: {before_rows}"
)

print(
    f"Rows after cleaning : {after_rows}"
)

print(
    f"Rows removed        : {before_rows - after_rows}"
)


# ============================================================
# 8. VALIDATE TARGET
# ============================================================

data[TARGET] = data[TARGET].astype(int)


actual_labels = set(
    data[TARGET].unique()
)


valid_labels = {
    0,
    1,
    2
}


if not actual_labels.issubset(valid_labels):

    raise ValueError(

        f"Invalid target labels found: {actual_labels}\n"

        f"Expected only: {valid_labels}"

    )


# ============================================================
# 9. FEATURE ENGINEERING
# ============================================================

print("\n============================================================")
print("FEATURE ENGINEERING")
print("============================================================")


Q_COLUMNS = [

    "Q1",
    "Q2",
    "Q3",
    "Q4",
    "Q5",
    "Q6",
    "Q7",
    "Q8",
    "Q9",
    "Q10"

]


# ------------------------------------------------------------
# Survey statistics
# ------------------------------------------------------------

data["Q_Mean"] = data[
    Q_COLUMNS
].mean(axis=1)


data["Q_Total"] = data[
    Q_COLUMNS
].sum(axis=1)


data["Q_Std"] = data[
    Q_COLUMNS
].std(axis=1)


data["Q_Min"] = data[
    Q_COLUMNS
].min(axis=1)


data["Q_Max"] = data[
    Q_COLUMNS
].max(axis=1)


# ------------------------------------------------------------
# Expected yield
# ------------------------------------------------------------

data["Expected_Yield"] = (

    data["Farm Size (ha)"]

    *

    data["Average Yield (tons/ha)"]

)


# ------------------------------------------------------------
# Yield difference
# ------------------------------------------------------------

data["Yield_Difference"] = (

    data["Crop Yield (tons)"]

    -

    data["Expected_Yield"]

)


# ------------------------------------------------------------
# Actual yield per hectare
# ------------------------------------------------------------

data["Actual_Yield_Per_Ha"] = np.where(

    data["Farm Size (ha)"] > 0,

    data["Crop Yield (tons)"]
    /
    data["Farm Size (ha)"],

    0

)


# ------------------------------------------------------------
# Yield efficiency
# ------------------------------------------------------------

data["Yield_Efficiency"] = np.where(

    data["Expected_Yield"] > 0,

    data["Crop Yield (tons)"]
    /
    data["Expected_Yield"],

    0

)


# ------------------------------------------------------------
# Subsidy per hectare
# ------------------------------------------------------------

data["Subsidy_per_Ha"] = np.where(

    data["Farm Size (ha)"] > 0,

    data["Subsidy Received"]
    /
    data["Farm Size (ha)"],

    0

)


# ============================================================
# 10. FINAL FEATURES
# ============================================================

FEATURES = [

    # Agricultural variables

    "Farm Size (ha)",

    "Average Yield (tons/ha)",

    "Crop Yield (tons)",

    "Average Selling Price (₱/kg)",

    "Subsidy Received",


    # Survey variables

    "Q1",
    "Q2",
    "Q3",
    "Q4",
    "Q5",
    "Q6",
    "Q7",
    "Q8",
    "Q9",
    "Q10",


    # Survey aggregates

    "Q_Mean",

    "Q_Total",

    "Q_Std",

    "Q_Min",

    "Q_Max",


    # Yield features

    "Expected_Yield",

    "Yield_Difference",

    "Actual_Yield_Per_Ha",

    "Yield_Efficiency",


    # Subsidy feature

    "Subsidy_per_Ha"

]


# ============================================================
# 11. REMOVE INFINITE VALUES
# ============================================================

data = data.replace(
    [np.inf, -np.inf],
    np.nan
)


data = data.dropna(
    subset=FEATURES + [TARGET]
)


# ============================================================
# 12. PREPARE X AND Y
# ============================================================

X = data[
    FEATURES
].copy()


y = data[
    TARGET
].copy()


# ============================================================
# 13. TARGET DISTRIBUTION
# ============================================================

print("\n============================================================")
print("TARGET DISTRIBUTION")
print("============================================================")


target_distribution = (

    y.value_counts()

    .sort_index()

    .rename(

        index={

            0: "Not Effective",

            1: "Moderately Effective",

            2: "Effective"

        }

    )

)


print(
    target_distribution
)


# ============================================================
# 14. FIRST SPLIT
#
# 80% development
# 20% final TEST
#
# TEST WILL NOT BE USED FOR MODEL SELECTION
# ============================================================

X_development, X_test, y_development, y_test = (

    train_test_split(

        X,

        y,

        test_size=0.20,

        stratify=y,

        random_state=RANDOM_STATE

    )

)


# ============================================================
# 15. SECOND SPLIT
#
# 80% TRAIN
# 20% VALIDATION
#
# Overall:
#
# TRAIN      = 64%
# VALIDATION = 16%
# TEST       = 20%
# ============================================================

X_train, X_validation, y_train, y_validation = (

    train_test_split(

        X_development,

        y_development,

        test_size=0.20,

        stratify=y_development,

        random_state=RANDOM_STATE

    )

)


print("\n============================================================")
print("DATA SPLIT")
print("============================================================")


print(
    "Training samples   :", len(X_train)
)

print(
    "Validation samples :", len(X_validation)
)

print(
    "Test samples       :", len(X_test)
)


# ============================================================
# 16. CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=RANDOM_STATE

)


# ============================================================
# 17. RANDOM FOREST PIPELINE
# ============================================================

pipeline = Pipeline(

    steps=[

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

    ]

)


# ============================================================
# 18. HYPERPARAMETER SEARCH
#
# IMPORTANT:
# Search is performed ONLY on X_train/y_train.
#
# Validation and test remain unseen.
# ============================================================

print("\n============================================================")
print("HYPERPARAMETER SEARCH")
print("============================================================")


param_distributions = {

    # --------------------------------------------------------
    # SMOTE
    # --------------------------------------------------------

    "smote": [

        SMOTE(
            random_state=RANDOM_STATE
        ),

        "passthrough"

    ],


    # --------------------------------------------------------
    # Number of trees
    # --------------------------------------------------------

    "classifier__n_estimators": [

        50,
        75,
        100,
        150,
        200,
        250,
        300,
        400,
        500,
        600

    ],


    # --------------------------------------------------------
    # Maximum depth
    # --------------------------------------------------------

    "classifier__max_depth": [

        None,
        5,
        6,
        8,
        10,
        12,
        15,
        20,
        25

    ],


    # --------------------------------------------------------
    # Minimum samples split
    # --------------------------------------------------------

    "classifier__min_samples_split": [

        2,
        3,
        4,
        5,
        8,
        10,
        15

    ],


    # --------------------------------------------------------
    # Minimum samples leaf
    # --------------------------------------------------------

    "classifier__min_samples_leaf": [

        1,
        2,
        3,
        4,
        5,
        6,
        8

    ],


    # --------------------------------------------------------
    # Maximum features
    # --------------------------------------------------------

    "classifier__max_features": [

        "sqrt",
        "log2",
        None,
        0.5,
        0.7,
        0.8

    ],


    # --------------------------------------------------------
    # Bootstrap
    # --------------------------------------------------------

    # IMPORTANT:
    # Keep bootstrap=True because max_samples is used.

    "classifier__bootstrap": [

        True

    ],


    # --------------------------------------------------------
    # Class weighting
    # --------------------------------------------------------

    "classifier__class_weight": [

        None,
        "balanced",
        "balanced_subsample"

    ],


    # --------------------------------------------------------
    # Criterion
    # --------------------------------------------------------

    "classifier__criterion": [

        "gini",
        "entropy",
        "log_loss"

    ],


    # --------------------------------------------------------
    # Maximum samples
    # --------------------------------------------------------

    "classifier__max_samples": [

        None,
        0.7,
        0.8,
        0.9

    ]

}


# ============================================================
# 19. RANDOMIZED SEARCH
# ============================================================

search = RandomizedSearchCV(

    estimator=pipeline,

    param_distributions=param_distributions,

    n_iter=100,

    scoring="accuracy",

    cv=cv,

    verbose=1,

    random_state=RANDOM_STATE,

    n_jobs=-1,

    return_train_score=True

)


search.fit(

    X_train,

    y_train

)


best_model = search.best_estimator_


# ============================================================
# 20. BEST PARAMETERS
# ============================================================

print("\n============================================================")
print("BEST HYPERPARAMETERS")
print("============================================================")


for key, value in search.best_params_.items():

    print(
        f"{key}: {value}"
    )


print(
    "\nBest Training CV Accuracy:",
    round(
        search.best_score_,
        4
    )
)


# ============================================================
# 21. VALIDATION PERFORMANCE
# ============================================================

validation_pred = best_model.predict(

    X_validation

)


validation_accuracy = accuracy_score(

    y_validation,

    validation_pred

)


validation_f1 = f1_score(

    y_validation,

    validation_pred,

    average="macro",

    zero_division=0

)


print("\n============================================================")
print("VALIDATION PERFORMANCE")
print("============================================================")


print(
    f"Validation Accuracy : {validation_accuracy:.4f}"
)

print(
    f"Validation Macro F1 : {validation_f1:.4f}"
)


# ============================================================
# 22. FINAL MODEL
#
# After model selection is complete:
#
# TRAIN + VALIDATION are combined.
#
# TEST remains completely untouched.
# ============================================================

X_final_train = pd.concat(

    [
        X_train,
        X_validation
    ],

    axis=0

)


y_final_train = pd.concat(

    [
        y_train,
        y_validation
    ],

    axis=0

)


print("\n============================================================")
print("FINAL MODEL TRAINING")
print("============================================================")


print(
    "Final training samples:",
    len(X_final_train)
)


# ------------------------------------------------------------
# Extract selected parameters
# ------------------------------------------------------------

selected_smote = search.best_params_["smote"]


selected_classifier_params = {

    "n_estimators":
        search.best_params_[
            "classifier__n_estimators"
        ],

    "max_depth":
        search.best_params_[
            "classifier__max_depth"
        ],

    "min_samples_split":
        search.best_params_[
            "classifier__min_samples_split"
        ],

    "min_samples_leaf":
        search.best_params_[
            "classifier__min_samples_leaf"
        ],

    "max_features":
        search.best_params_[
            "classifier__max_features"
        ],

    "bootstrap":
        search.best_params_[
            "classifier__bootstrap"
        ],

    "class_weight":
        search.best_params_[
            "classifier__class_weight"
        ],

    "criterion":
        search.best_params_[
            "classifier__criterion"
        ],

    "max_samples":
        search.best_params_[
            "classifier__max_samples"
        ],

    "random_state":
        RANDOM_STATE,

    "n_jobs":
        -1

}


final_steps = []


if selected_smote != "passthrough":

    final_steps.append(

        (

            "smote",

            SMOTE(

                random_state=RANDOM_STATE

            )

        )

    )


final_steps.append(

    (

        "classifier",

        RandomForestClassifier(

            **selected_classifier_params

        )

    )

)


final_model = Pipeline(

    steps=final_steps

)


final_model.fit(

    X_final_train,

    y_final_train

)


# ============================================================
# 23. FINAL TRAINING ACCURACY
# ============================================================

final_train_pred = final_model.predict(

    X_final_train

)


final_train_accuracy = accuracy_score(

    y_final_train,

    final_train_pred

)


print(
    f"Final Training Accuracy: {final_train_accuracy:.4f}"
)


# ============================================================
# 24. FINAL TEST PREDICTION
# ============================================================

y_test_pred = final_model.predict(

    X_test

)


# ============================================================
# 25. TEST METRICS
# ============================================================

accuracy = accuracy_score(

    y_test,

    y_test_pred

)


precision = precision_score(

    y_test,

    y_test_pred,

    average="weighted",

    zero_division=0

)


recall = recall_score(

    y_test,

    y_test_pred,

    average="weighted",

    zero_division=0

)


f1_weighted = f1_score(

    y_test,

    y_test_pred,

    average="weighted",

    zero_division=0

)


f1_macro = f1_score(

    y_test,

    y_test_pred,

    average="macro",

    zero_division=0

)


# ============================================================
# 26. ORDINAL ERROR METRICS
# ============================================================

mae = mean_absolute_error(

    y_test,

    y_test_pred

)


mse = mean_squared_error(

    y_test,

    y_test_pred

)


rmse = np.sqrt(

    mse

)


r2 = r2_score(

    y_test,

    y_test_pred

)


# ============================================================
# 27. CROSS VALIDATION ON FINAL DEVELOPMENT DATA
#
# This is only for reporting stability.
# ============================================================

cv_accuracy_scores = cross_val_score(

    final_model,

    X_final_train,

    y_final_train,

    cv=cv,

    scoring="accuracy",

    n_jobs=-1

)


cv_f1_scores = cross_val_score(

    final_model,

    X_final_train,

    y_final_train,

    cv=cv,

    scoring="f1_macro",

    n_jobs=-1

)


mean_cv_accuracy = (

    cv_accuracy_scores.mean()

)


std_cv_accuracy = (

    cv_accuracy_scores.std()

)


mean_cv_f1 = (

    cv_f1_scores.mean()

)


std_cv_f1 = (

    cv_f1_scores.std()

)


# ============================================================
# 28. FINAL RESULTS
# ============================================================

print("\n============================================================")
print("FINAL TEST PERFORMANCE")
print("============================================================")


print(
    f"Training Accuracy   : {final_train_accuracy:.4f}"
)

print(
    f"Validation Accuracy : {validation_accuracy:.4f}"
)

print(
    f"Test Accuracy       : {accuracy:.4f}"
)

print(
    f"Weighted Precision  : {precision:.4f}"
)

print(
    f"Weighted Recall     : {recall:.4f}"
)

print(
    f"Weighted F1         : {f1_weighted:.4f}"
)

print(
    f"Macro F1            : {f1_macro:.4f}"
)

print(
    f"MAE                 : {mae:.4f}"
)

print(
    f"MSE                 : {mse:.4f}"
)

print(
    f"RMSE                : {rmse:.4f}"
)

print(
    f"R2                  : {r2:.4f}"
)

print(
    f"Mean CV Accuracy    : {mean_cv_accuracy:.4f}"
)

print(
    f"CV Accuracy Std     : {std_cv_accuracy:.4f}"
)

print(
    f"Mean CV Macro F1    : {mean_cv_f1:.4f}"
)

print(
    f"CV Macro F1 Std     : {std_cv_f1:.4f}"
)


# ============================================================
# 29. CLASSIFICATION REPORT
# ============================================================

print("\n============================================================")
print("CLASSIFICATION REPORT")
print("============================================================")


print(

    classification_report(

        y_test,

        y_test_pred,

        labels=CLASS_LABELS,

        target_names=CLASS_NAMES,

        digits=4,

        zero_division=0

    )

)


# ============================================================
# 30. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    y_test,

    y_test_pred,

    labels=CLASS_LABELS

)


plt.figure(

    figsize=(8, 6)

)


sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues",

    xticklabels=CLASS_NAMES,

    yticklabels=CLASS_NAMES

)


plt.title(

    "Confusion Matrix - Test Set",

    fontsize=14,

    fontweight="bold"

)


plt.xlabel(

    "Predicted Class"

)


plt.ylabel(

    "Actual Class"

)


plt.tight_layout()


plt.savefig(

    os.path.join(

        OUTPUT_DIR,

        "confusion_matrix_test.png"

    ),

    dpi=300,

    bbox_inches="tight"

)


plt.close()


# ============================================================
# 31. PER-CLASS METRICS
# ============================================================

per_class_results = []


total_test = len(y_test)


for class_id in CLASS_LABELS:

    TP = cm[
        class_id,
        class_id
    ]


    FN = (

        cm[
            class_id,
            :
        ].sum()

        -

        TP

    )


    FP = (

        cm[
            :,
            class_id
        ].sum()

        -

        TP

    )


    TN = (

        total_test

        -

        TP

        -

        FP

        -

        FN

    )


    class_accuracy = (

        (TP + TN)

        /

        total_test

    )


    class_precision = (

        TP

        /

        (TP + FP)

        if TP + FP > 0

        else 0

    )


    class_recall = (

        TP

        /

        (TP + FN)

        if TP + FN > 0

        else 0

    )


    class_specificity = (

        TN

        /

        (TN + FP)

        if TN + FP > 0

        else 0

    )


    class_f1 = (

        2

        *

        class_precision

        *

        class_recall

        /

        (

            class_precision

            +

            class_recall

        )

        if class_precision + class_recall > 0

        else 0

    )


    per_class_results.append({

        "Class":
            CLASS_NAMES[class_id],

        "TP":
            TP,

        "FP":
            FP,

        "FN":
            FN,

        "TN":
            TN,

        "Accuracy":
            class_accuracy,

        "Precision":
            class_precision,

        "Recall":
            class_recall,

        "F1-Score":
            class_f1,

        "Specificity":
            class_specificity

    })


per_class_df = pd.DataFrame(

    per_class_results

)


# ============================================================
# 32. PER-CLASS HEATMAP
# ============================================================

heatmap_df = (

    per_class_df

    .set_index(

        "Class"

    )

)


plt.figure(

    figsize=(15, 5)

)


sns.heatmap(

    heatmap_df,

    annot=True,

    fmt=".2f",

    cmap="Blues",

    linewidths=0.5

)


plt.title(

    "Per-Class Metrics (Test Set)",

    fontsize=14,

    fontweight="bold"

)


plt.xlabel(

    "Metrics"

)


plt.ylabel(

    "Class"

)


plt.xticks(

    rotation=0

)


plt.yticks(

    rotation=0

)


plt.tight_layout()


plt.savefig(

    os.path.join(

        OUTPUT_DIR,

        "per_class_metrics_test.png"

    ),

    dpi=300,

    bbox_inches="tight"

)


plt.close()


# ============================================================
# 33. RANDOM FOREST PERFORMANCE VS NUMBER OF TREES
#
# IMPORTANT:
# KEEP THIS GRAPH LAYOUT THE SAME.
#
# Train Accuracy
# Validation Accuracy
# Test Accuracy
#
# Test accuracy is displayed for comparison only.
# It is NOT used to select the number of trees.
# ============================================================

tree_values = [

    10,
    25,
    50,
    75,
    100,
    150,
    200

]


train_accuracy_values = []

validation_accuracy_values = []

test_accuracy_values = []


# ============================================================
# 34. TREE EXPERIMENT
# ============================================================

for n_trees in tree_values:

    tree_steps = []


    # --------------------------------------------------------
    # SMOTE
    # --------------------------------------------------------

    if selected_smote != "passthrough":

        tree_steps.append(

            (

                "smote",

                SMOTE(

                    random_state=RANDOM_STATE

                )

            )

        )


    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    tree_params = (

        selected_classifier_params.copy()

    )


    tree_params[
        "n_estimators"
    ] = n_trees


    tree_steps.append(

        (

            "classifier",

            RandomForestClassifier(

                **tree_params

            )

        )

    )


    tree_model = Pipeline(

        steps=tree_steps

    )


    # --------------------------------------------------------
    # Train ONLY on the training split.
    #
    # Validation and test remain unseen.
    # --------------------------------------------------------

    tree_model.fit(

        X_train,

        y_train

    )


    train_pred = tree_model.predict(

        X_train

    )


    validation_pred = tree_model.predict(

        X_validation

    )


    test_pred = tree_model.predict(

        X_test

    )


    train_accuracy_values.append(

        accuracy_score(

            y_train,

            train_pred

        )

    )


    validation_accuracy_values.append(

        accuracy_score(

            y_validation,

            validation_pred

        )

    )


    test_accuracy_values.append(

        accuracy_score(

            y_test,

            test_pred

        )

    )


# ============================================================
# 35. TREE RESULTS EXCEL
# ============================================================

tree_results = pd.DataFrame({

    "Number of Trees":
        tree_values,

    "Train Accuracy":
        train_accuracy_values,

    "Validation Accuracy":
        validation_accuracy_values,

    "Test Accuracy":
        test_accuracy_values

})


tree_results.to_excel(

    os.path.join(

        OUTPUT_DIR,

        "random_forest_tree_performance.xlsx"

    ),

    index=False

)


# ============================================================
# 36. RANDOM FOREST PERFORMANCE GRAPH
#
# SAME LAYOUT AS YOUR SCREENSHOT
# ============================================================

plt.figure(

    figsize=(10, 6)

)


# ------------------------------------------------------------
# TRAIN ACCURACY
# ------------------------------------------------------------

plt.plot(

    tree_values,

    train_accuracy_values,

    marker="o",

    linewidth=2,

    label="Train Accuracy"

)


# ------------------------------------------------------------
# VALIDATION ACCURACY
# ------------------------------------------------------------

plt.plot(

    tree_values,

    validation_accuracy_values,

    marker="s",

    linewidth=2,

    label="Validation Accuracy"

)


# ------------------------------------------------------------
# TEST ACCURACY
# ------------------------------------------------------------

plt.plot(

    tree_values,

    test_accuracy_values,

    marker="^",

    linewidth=2,

    label="Test Accuracy"

)


# ------------------------------------------------------------
# TITLE
# ------------------------------------------------------------

plt.title(

    "Random Forest Performance vs Number of Trees",

    fontsize=14,

    fontweight="bold"

)


# ------------------------------------------------------------
# X AXIS
# ------------------------------------------------------------

plt.xlabel(

    "Number of Trees (n_estimators)"

)


# ------------------------------------------------------------
# Y AXIS
# ------------------------------------------------------------

plt.ylabel(

    "Accuracy"

)


# ------------------------------------------------------------
# SAME SCALE STYLE
# ------------------------------------------------------------

plt.ylim(

    0.0,

    1.0

)


# ------------------------------------------------------------
# SAME TREE VALUES
# ------------------------------------------------------------

plt.xticks(

    tree_values

)


# ------------------------------------------------------------
# GRID
# ------------------------------------------------------------

plt.grid(

    True,

    alpha=0.3

)


# ------------------------------------------------------------
# LEGEND
# ------------------------------------------------------------

plt.legend()


# ------------------------------------------------------------
# LAYOUT
# ------------------------------------------------------------

plt.tight_layout()


# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

plt.savefig(

    os.path.join(

        OUTPUT_DIR,

        "random_forest_performance_vs_trees.png"

    ),

    dpi=300,

    bbox_inches="tight"

)


plt.close()


# ============================================================
# 37. FEATURE IMPORTANCE
# ============================================================

print(
    "\nCalculating feature importance..."
)


rf = final_model.named_steps[

    "classifier"

]


feature_importance_df = pd.DataFrame({

    "Feature":

        FEATURES,

    "Importance":

        rf.feature_importances_

})


feature_importance_df = (

    feature_importance_df

    .sort_values(

        "Importance",

        ascending=False

    )

    .reset_index(

        drop=True

    )

)


feature_importance_df[

    "Importance (%)"

] = (

    feature_importance_df[

        "Importance"

    ]

    *

    100

)


feature_importance_df.to_excel(

    os.path.join(

        OUTPUT_DIR,

        "feature_importance.xlsx"

    ),

    index=False

)


# ============================================================
# 38. PERMUTATION IMPORTANCE
# ============================================================

print(
    "\nCalculating permutation importance..."
)


permutation = permutation_importance(

    final_model,

    X_test,

    y_test,

    n_repeats=10,

    random_state=RANDOM_STATE,

    scoring="accuracy",

    n_jobs=-1

)


permutation_df = pd.DataFrame({

    "Feature":

        FEATURES,

    "Importance Mean":

        permutation.importances_mean,

    "Importance Std":

        permutation.importances_std

})


permutation_df = (

    permutation_df

    .sort_values(

        "Importance Mean",

        ascending=False

    )

    .reset_index(

        drop=True

    )

)


permutation_df.to_excel(

    os.path.join(

        OUTPUT_DIR,

        "permutation_importance.xlsx"

    ),

    index=False

)


# ============================================================
# 39. TRAINING METRICS EXCEL
# ============================================================

training_metrics = pd.DataFrame({

    "Metric": [

        "Training Accuracy",

        "Validation Accuracy",

        "Test Accuracy",

        "Weighted Precision",

        "Weighted Recall",

        "Weighted F1",

        "Macro F1",

        "MAE",

        "MSE",

        "RMSE",

        "R2",

        "Mean CV Accuracy",

        "CV Accuracy Std",

        "Mean CV Macro F1",

        "CV Macro F1 Std"

    ],

    "Value": [

        final_train_accuracy,

        validation_accuracy,

        accuracy,

        precision,

        recall,

        f1_weighted,

        f1_macro,

        mae,

        mse,

        rmse,

        r2,

        mean_cv_accuracy,

        std_cv_accuracy,

        mean_cv_f1,

        std_cv_f1

    ]

})


training_metrics.to_excel(

    os.path.join(

        OUTPUT_DIR,

        "training_metrics.xlsx"

    ),

    index=False

)


# ============================================================
# 40. PER-CLASS METRICS EXCEL
# ============================================================

per_class_df.to_excel(

    os.path.join(

        OUTPUT_DIR,

        "per_class_metrics.xlsx"

    ),

    index=False

)


# ============================================================
# 41. CLASSIFICATION REPORT EXCEL
# ============================================================

report = classification_report(

    y_test,

    y_test_pred,

    labels=CLASS_LABELS,

    target_names=CLASS_NAMES,

    output_dict=True,

    zero_division=0

)


classification_df = pd.DataFrame(

    report

).transpose()


classification_df.to_excel(

    os.path.join(

        OUTPUT_DIR,

        "classification_report.xlsx"

    )

)


# ============================================================
# 42. SAVE FINAL MODEL
# ============================================================

joblib.dump(

    final_model,

    os.path.join(

        OUTPUT_DIR,

        "random_forest_subsidy.pkl"

    )

)


# ============================================================
# 43. SAVE MODEL FEATURES
# ============================================================

joblib.dump(

    FEATURES,

    os.path.join(

        OUTPUT_DIR,

        "model_features.pkl"

    )

)


# ============================================================
# 44. SAVE BEST PARAMETERS
# ============================================================

best_parameters_df = pd.DataFrame({

    "Parameter":

        list(search.best_params_.keys()),

    "Value":

        [

            str(value)

            for value

            in search.best_params_.values()

        ]

})


best_parameters_df.to_excel(

    os.path.join(

        OUTPUT_DIR,

        "best_hyperparameters.xlsx"

    ),

    index=False

)


# ============================================================
# 45. FINAL OUTPUT
# ============================================================

print("\n============================================================")
print("TRAINING COMPLETE")
print("============================================================")


print("\nBEST CONFIGURATION:")

for key, value in search.best_params_.items():

    print(
        f"{key}: {value}"
    )


print("\n============================================================")
print("FINAL RESULTS")
print("============================================================")


print(
    f"Training Accuracy   : {final_train_accuracy:.4f}"
)

print(
    f"Validation Accuracy : {validation_accuracy:.4f}"
)

print(
    f"Test Accuracy       : {accuracy:.4f}"
)

print(
    f"Weighted Precision  : {precision:.4f}"
)

print(
    f"Weighted Recall     : {recall:.4f}"
)

print(
    f"Weighted F1         : {f1_weighted:.4f}"
)

print(
    f"Macro F1            : {f1_macro:.4f}"
)

print(
    f"MAE                 : {mae:.4f}"
)

print(
    f"MSE                 : {mse:.4f}"
)

print(
    f"RMSE                : {rmse:.4f}"
)

print(
    f"R2                  : {r2:.4f}"
)

print(
    f"Mean CV Accuracy    : {mean_cv_accuracy:.4f}"
)

print(
    f"CV Accuracy Std     : {std_cv_accuracy:.4f}"
)

print(
    f"Mean CV Macro F1    : {mean_cv_f1:.4f}"
)

print(
    f"CV Macro F1 Std     : {std_cv_f1:.4f}"
)


# ============================================================
# 46. GENERATED FILES
# ============================================================

print("\n============================================================")
print("GENERATED FILES")
print("============================================================")


files = [

    "random_forest_subsidy.pkl",

    "model_features.pkl",

    "confusion_matrix_test.png",

    "per_class_metrics_test.png",

    "random_forest_performance_vs_trees.png",

    "random_forest_tree_performance.xlsx",

    "feature_importance.xlsx",

    "permutation_importance.xlsx",

    "training_metrics.xlsx",

    "per_class_metrics.xlsx",

    "classification_report.xlsx",

    "best_hyperparameters.xlsx"

]


for file in files:

    print(

        " -",

        os.path.join(

            OUTPUT_DIR,

            file

        )

    )


print("\n============================================================")
print("DONE")
print("============================================================")