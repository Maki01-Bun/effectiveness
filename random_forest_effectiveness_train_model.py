import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    learning_curve,
    cross_val_score
)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_excel("datasets/subsidy_dataset.xlsx")

print("\n==============================")
print("DATASET")
print("==============================")
print("Dataset Shape:", df.shape)
print(df.head())

# =====================================================
# FEATURES AND TARGET
# =====================================================

X = df.drop(columns=["Effectiveness Label"])
y = df["Effectiveness Label"]

# =====================================================
# CATEGORICAL & NUMERICAL FEATURES
# =====================================================

categorical_cols = [
    "Subsidy Type",
    "Pest",
    "Calamity"
]

numerical_cols = [
    "Farm Size (ha)",
    "Crop Yield Before",
    "Crop Yield After",
    "Income Before",
    "Income After",
    "Feedback Score"
]

# =====================================================
# PREPROCESSOR
# =====================================================

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

# =====================================================
# RANDOM FOREST (IMPROVED)
# =====================================================

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

# =====================================================
# PIPELINE
# =====================================================

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", rf_model)
])

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# =====================================================
# TRAIN MODEL
# =====================================================

pipeline.fit(X_train, y_train)

# =====================================================
# TRAIN VS TEST ACCURACY
# =====================================================

train_accuracy = pipeline.score(X_train, y_train)
test_accuracy = pipeline.score(X_test, y_test)

print("\n==============================")
print("TRAIN VS TEST ACCURACY")
print("==============================")
print(f"Train Accuracy : {train_accuracy * 100:.2f}%")
print(f"Test Accuracy  : {test_accuracy * 100:.2f}%")

# =====================================================
# PREDICTIONS
# =====================================================

y_pred = pipeline.predict(X_test)

# =====================================================
# ACCURACY
# =====================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("ACCURACY")
print("==============================")
print(f"Accuracy: {accuracy * 100:.2f}%")

# =====================================================
# CLASSIFICATION REPORT
# =====================================================

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")
print(classification_report(y_test, y_pred))

# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(y_test, y_pred)

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")
print(cm)

# =====================================================
# SAVE CONFUSION MATRIX IMAGE
# =====================================================

fig_cm, ax_cm = plt.subplots(figsize=(7, 6))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=pipeline.classes_
)

disp.plot(ax=ax_cm)

plt.title("Random Forest Confusion Matrix")

plt.savefig(
    "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =====================================================
# LEARNING CURVE
# =====================================================

train_sizes, train_scores, test_scores = learning_curve(
    pipeline,
    X,
    y,
    cv=5,
    scoring="accuracy",
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)

# =====================================================
# CROSS VALIDATION
# =====================================================

cv_scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("\n==============================")
print("CROSS VALIDATION")
print("==============================")
print("Scores:", cv_scores)
print(f"Mean Accuracy: {cv_scores.mean()*100:.2f}%")
print(f"Std Dev: {cv_scores.std()*100:.2f}%")

# =====================================================
# TRAINING RESULTS FIGURE
# =====================================================

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

fig.suptitle(
    "Random Forest Training Results",
    fontsize=16,
    fontweight="bold"
)

# =====================================================
# LEARNING CURVE
# =====================================================

axes[0].plot(
    train_sizes,
    train_mean,
    marker='o',
    linewidth=2,
    label='Training Accuracy'
)

axes[0].plot(
    train_sizes,
    test_mean,
    marker='o',
    linewidth=2,
    label='Validation Accuracy'
)

axes[0].set_title("Learning Curve")
axes[0].set_xlabel("Training Samples")
axes[0].set_ylabel("Accuracy")
axes[0].set_ylim(0.80, 1.01)
axes[0].legend()
axes[0].grid(True)

# =====================================================
# CROSS VALIDATION ACCURACY
# =====================================================

folds = [f"Fold {i}" for i in range(1, 6)]

bars = axes[1].bar(
    folds,
    cv_scores * 100
)

axes[1].set_title("Cross-Validation Accuracy")
axes[1].set_ylabel("Accuracy (%)")
axes[1].set_ylim(0, 100)

for bar, score in zip(bars, cv_scores):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        score * 100 + 1,
        f"{score * 100:.2f}%",
        ha='center'
    )

# =====================================================
# CONFUSION MATRIX
# =====================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=pipeline.classes_
)

disp.plot(
    ax=axes[2],
    colorbar=False
)

axes[2].set_title("Confusion Matrix")

# =====================================================
# SAVE FIGURE
# =====================================================

plt.tight_layout()

plt.savefig(
    "training_results.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =====================================================
# SAVE MODEL
# =====================================================

joblib.dump(
    pipeline,
    "random_forest_subsidy.pkl"
)

print("\n==============================")
print("FILES GENERATED")
print("==============================")
print("✓ random_forest_subsidy.pkl")
print("✓ confusion_matrix.png")
print("✓ training_results.png")
print("==============================")