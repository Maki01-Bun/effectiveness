import random
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


random.seed(42)
np.random.seed(42)

TRAIN_FILE = "datasets/subsidy_dataset.xlsx"
VALIDATION_FILE = "datasets/subsidy_validation_datasets.xlsx"
TARGET = "Effectiveness Label"


# LOAD DATASET
train_df = pd.read_excel(TRAIN_FILE)
validation_df = pd.read_excel(VALIDATION_FILE)

print("\n==============================")
print("TRAINING DATASET")
print("==============================")
print(train_df.head())
print("Shape:", train_df.shape)

# FEATURES AND TARGET
X = train_df.drop(columns=[TARGET])
y = train_df[TARGET]

X_validation = validation_df.drop(columns=[TARGET])
y_validation = validation_df[TARGET]

categorical_cols = ["Subsidy Type", "Pest", "Calamity"]
numerical_cols = [
    "Farm Size (ha)",
    "Crop Yield Before",
    "Crop Yield After",
    "Feedback Score",
]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", "passthrough", numerical_cols),
])


# TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)


# MODEL + GRID SEARCH
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42, n_jobs=-1))
])  
params = {
    "classifier__n_estimators": [100, 200, 300, 500],
    "classifier__max_depth": [None, 10, 15, 20, 30],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
    "classifier__max_features": ["sqrt", "log2"],
    "classifier__bootstrap": [True, False],
}

search = RandomizedSearchCV(
    pipeline,
    param_distributions=params,
    n_iter=40,
    cv=5,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1,
)

search.fit(X_train, y_train)
model = search.best_estimator_

print("Best Parameters:", search.best_params_)


# TRAIN / TEST EVALUATION
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print(f"Train Accuracy : {train_acc*100:.2f}%")
print(f"Test Accuracy  : {test_acc*100:.2f}%")

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=model.classes_).plot()
plt.savefig("confusion_matrix_test.png", dpi=300, bbox_inches="tight")
plt.close()


# CROSS VALIDATION
cv = cross_val_score(model, X, y, cv=5, scoring="accuracy", n_jobs=-1)
print("CV Scores:", cv)
print("CV Mean:", cv.mean())

# LEARNING CURVE
sizes, train_scores, valid_scores = learning_curve(
    model, X, y,
    cv=5,
    train_sizes=np.linspace(0.1,1.0,10),
    n_jobs=-1
)

plt.figure(figsize=(8,5))
plt.plot(sizes, train_scores.mean(axis=1), marker="o", label="Training")
plt.plot(sizes, valid_scores.mean(axis=1), marker="o", label="Validation")
plt.xlabel("Training Samples")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.savefig("learning_curve.png", dpi=300, bbox_inches="tight")
plt.close()


# EXTERNAL VALIDATION
val_pred = model.predict(X_validation)
val_acc = accuracy_score(y_validation, val_pred)

print(f"External Validation Accuracy : {val_acc*100:.2f}%")
print(classification_report(y_validation, val_pred))

val_cm = confusion_matrix(y_validation, val_pred)
ConfusionMatrixDisplay(val_cm, display_labels=model.classes_).plot()
plt.savefig("confusion_matrix_validation.png", dpi=300, bbox_inches="tight")
plt.close()


# FEATURE IMPORTANCE
feature_names = model.named_steps["preprocessor"].get_feature_names_out()
feature_importance = model.named_steps["classifier"].feature_importances_

print("Number of Features:", len(feature_names))
print("Number of Importances:", len(feature_importance))

fi = pd.DataFrame({
    "Feature": feature_names,
    "Importance": feature_importance
})

fi = fi.sort_values(by="Importance", ascending=False)

fi.to_excel("feature_importance.xlsx", index=False)

print("\nTop 10 Most Important Features")
print(fi.head(10))

# METRICS EXPORT
metrics = pd.DataFrame({
    "Metric":[
        "Train Accuracy",
        "Test Accuracy",
        "Validation Accuracy",
        "Cross Validation"
    ],
    "Value":[train_acc, test_acc, val_acc, cv.mean()]
})

metrics.to_excel("training_metrics.xlsx", index=False)


# SAVE MODEL
joblib.dump(model, "random_forest_subsidy.pkl")

print("\nTraining Complete!")
print("Generated Files:")
print("- random_forest_subsidy.pkl")
print("- confusion_matrix_test.png")
print("- confusion_matrix_validation.png")  
print("- learning_curve.png")
print("- feature_importance.xlsx")
print("- training_metrics.xlsx")