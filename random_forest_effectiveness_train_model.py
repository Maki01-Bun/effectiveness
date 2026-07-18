import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
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


# LOAD DATASET

df = pd.read_excel("datasets/subsidy_dataset.xlsx")

print("Dataset Shape:", df.shape)
print(df.head())

# FEATURES AND TARGET

X = df.drop(columns=["Effectiveness Label"])
y = df["Effectiveness Label"]


# CATEGORICAL & NUMERICAL COLUMNS

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


# PREPROCESSING

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

# RANDOM FOREST MODEL

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# PIPELINE

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", rf_model)
])


# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# TRAIN MODEL

pipeline.fit(X_train, y_train)


# PREDICTIONS
y_pred = pipeline.predict(X_test)

# ACCURACY

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("ACCURACY")
print("==============================")
print(f"Accuracy: {accuracy * 100:.2f}%")

# ==========================================
# CLASSIFICATION REPORT
# ==========================================
print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")
print(classification_report(y_test, y_pred))


# CONFUSION MATRIX

cm = confusion_matrix(y_test, y_pred)

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")
print(cm)


# DISPLAY CONFUSION MATRIX

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=pipeline.classes_
)

fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax)

plt.title("Random Forest Confusion Matrix")

# Save Image
plt.savefig("confusion_matrix.png", dpi=300)

plt.show()


# FEATURE IMPORTANCE

print("\n==============================")
print("FEATURE IMPORTANCE")
print("==============================")

feature_names = (
    pipeline.named_steps["preprocessor"]
    .get_feature_names_out()
)

importances = (
    pipeline.named_steps["classifier"]
    .feature_importances_
)

feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

feature_importance_df = feature_importance_df.sort_values(
    by="Importance",
    ascending=False
)

print(feature_importance_df)


# SAVE MODEL

joblib.dump(
    pipeline,
    "random_forest_subsidy.pkl"
)

print("\nModel saved as random_forest_subsidy.pkl")
print("Confusion matrix saved as confusion_matrix.png")