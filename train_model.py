"""
train_model.py
---------------
Trains two models on the student performance dataset:

1. RandomForestRegressor  -> predicts the exact final_score (0-100)
2. RandomForestClassifier -> predicts the letter grade (A-F)

Saves evaluation metrics, a feature-importance chart, a
predicted-vs-actual chart, and a confusion matrix to outputs/,
and pickles both trained models to models/.

Run:
    python src/train_model.py
"""

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "student_performance.csv"
OUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"

NUMERIC_FEATURES = [
    "study_hours_per_week",
    "attendance_rate",
    "sleep_hours",
    "previous_gpa",
    "extracurricular_hours",
]
CATEGORICAL_FEATURES = ["gender", "parental_support", "internet_access", "study_group"]
RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["sleep_hours"] = df["sleep_hours"].fillna(df["sleep_hours"].median())
    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def train_regressor(df: pd.DataFrame) -> Pipeline:
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["final_score"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    pipeline = Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", RandomForestRegressor(n_estimators=300, max_depth=8, random_state=RANDOM_STATE)),
    ])
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("\n--- Regression: predicting final_score ---")
    print(f"MAE:  {mae:.2f} points")
    print(f"RMSE: {rmse:.2f} points")
    print(f"R^2:  {r2:.3f}")

    # Predicted vs actual plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, preds, alpha=0.5, color="#4C72B0")
    lims = [0, 100]
    plt.plot(lims, lims, "r--", label="Perfect prediction")
    plt.xlabel("Actual Score")
    plt.ylabel("Predicted Score")
    plt.title("Predicted vs Actual Final Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "06_predicted_vs_actual.png", dpi=150)
    plt.close()

    # Feature importance
    ohe_cols = pipeline.named_steps["preprocess"].named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES)
    all_features = NUMERIC_FEATURES + list(ohe_cols)
    importances = pipeline.named_steps["model"].feature_importances_
    imp_series = pd.Series(importances, index=all_features).sort_values(ascending=True)

    plt.figure(figsize=(8, 6))
    imp_series.tail(12).plot(kind="barh", color="#55A868")
    plt.title("Top Feature Importances (Regressor)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "07_feature_importance.png", dpi=150)
    plt.close()

    return pipeline


def train_classifier(df: pd.DataFrame) -> Pipeline:
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["grade"]
    # Note: not stratifying because the rare "F" grade bucket can have too
    # few samples for sklearn's stratified split to work reliably.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    pipeline = Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", RandomForestClassifier(n_estimators=300, max_depth=8, random_state=RANDOM_STATE)),
    ])
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print("\n--- Classification: predicting letter grade ---")
    print(f"Accuracy: {acc:.3f}")
    print(classification_report(y_test, preds, zero_division=0))

    labels = sorted(y.unique())
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, preds, labels=labels, cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix - Grade Classifier")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "08_confusion_matrix.png", dpi=150)
    plt.close()

    return pipeline


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data()

    reg_pipeline = train_regressor(data)
    clf_pipeline = train_classifier(data)

    joblib.dump(reg_pipeline, MODEL_DIR / "score_regressor.pkl")
    joblib.dump(clf_pipeline, MODEL_DIR / "grade_classifier.pkl")
    print(f"\nSaved trained models to {MODEL_DIR}")
