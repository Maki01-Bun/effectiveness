import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# LOAD DATASET
# ==========================================
df = pd.read_excel("datasets/subsidy_dataset.xlsx")

print("Dataset Shape:", df.shape)

# ==========================================
# FEATURES (X)
# ==========================================
X = df.drop(columns=[
    "Beneficiary ID",
    "Effectiveness Label"
])

# ==========================================
# TARGET (Y)
# ==========================================
y = df["Effectiveness Label"]

# ==========================================
# CATEGORICAL COLUMNS
# ==========================================
categorical_cols = [
    "Subsidy Type",
    "Pest/Calamity"
]

# ==========================================
# NUMERICAL COLUMNS
# ==========================================
numerical_cols = [
    "Farm Size (ha)",
    "Crop Yield Before",
    "Crop Yield After",
    "Income Before",
    "Income After",
    "Feedback Score"
]

# ==========================================
# PREPROCESSOR
# ==========================================
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

# ==========================================
# RANDOM FOREST
# ==========================================
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# ==========================================
# PIPELINE
# ==========================================
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", rf)
])

# ==========================================
# TRAIN TEST SPLIT
# 80% TRAINING
# 20% TESTING
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Records:", len(X_train))
print("Testing Records :", len(X_test))

# ==========================================
# TRAIN MODEL
# ==========================================
pipeline.fit(X_train, y_train)

# ==========================================
# PREDICT TEST DATA
# ==========================================
y_pred = pipeline.predict(X_test)

# ==========================================
# ACCURACY
# ==========================================
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

# ==========================================
# CLASSIFICATION REPORT
# ==========================================
print("\nClassification Report")
print(classification_report(y_test, y_pred))

# ==========================================
# CONFUSION MATRIX
# ==========================================
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix")
print(cm)

# ==========================================
# SAVE MODEL
# ==========================================
joblib.dump(
    pipeline,
    "random_forest_subsidy.pkl"
)

print("\nModel Saved Successfully!")