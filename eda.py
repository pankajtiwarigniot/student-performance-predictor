"""
eda.py
------
Exploratory Data Analysis for the student performance dataset.
Cleans the data, prints summary statistics, and saves a set of
matplotlib/seaborn charts to the outputs/ folder.

Run:
    python src/eda.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "student_performance.csv"
OUT_DIR = BASE_DIR / "outputs"

sns.set_theme(style="whitegrid")


def load_clean_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    # Handle missing values: impute sleep_hours with median (robust to skew)
    df["sleep_hours"] = df["sleep_hours"].fillna(df["sleep_hours"].median())

    # Sanity-check for duplicate students
    df = df.drop_duplicates(subset="student_id")

    return df


def print_summary(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Rows: {len(df)}  Columns: {df.shape[1]}")
    print("\nMissing values per column:\n", df.isna().sum())
    print("\nNumeric summary:\n", df.describe().round(2))
    print("\nGrade distribution:\n", df["grade"].value_counts())
    print("\nCorrelation with final_score:")
    numeric_cols = df.select_dtypes(include=np.number).columns.drop(["student_id", "final_score"])
    corr = df[numeric_cols.tolist() + ["final_score"]].corr()["final_score"].drop("final_score")
    print(corr.sort_values(ascending=False).round(3))


def make_plots(df: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Distribution of final scores
    plt.figure(figsize=(8, 5))
    sns.histplot(df["final_score"], bins=25, kde=True, color="#4C72B0")
    plt.title("Distribution of Final Scores")
    plt.xlabel("Final Score")
    plt.ylabel("Number of Students")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "01_score_distribution.png", dpi=150)
    plt.close()

    # 2. Study hours vs final score, colored by grade
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="study_hours_per_week", y="final_score", hue="grade",
                     palette="viridis", alpha=0.7)
    plt.title("Study Hours per Week vs Final Score")
    plt.xlabel("Study Hours / Week")
    plt.ylabel("Final Score")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "02_study_hours_vs_score.png", dpi=150)
    plt.close()

    # 3. Correlation heatmap
    plt.figure(figsize=(8, 6))
    numeric_cols = df.select_dtypes(include=np.number).drop(columns=["student_id"])
    sns.heatmap(numeric_cols.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "03_correlation_heatmap.png", dpi=150)
    plt.close()

    # 4. Grade distribution by parental support
    plt.figure(figsize=(8, 5))
    order = ["F", "D", "C", "B", "A"]
    sns.countplot(data=df, x="grade", hue="parental_support", order=order, palette="pastel")
    plt.title("Grade Distribution by Parental Support Level")
    plt.xlabel("Grade")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "04_grade_by_support.png", dpi=150)
    plt.close()

    # 5. Attendance vs score, with regression line
    plt.figure(figsize=(8, 5))
    sns.regplot(data=df, x="attendance_rate", y="final_score",
                scatter_kws={"alpha": 0.4}, line_kws={"color": "red"})
    plt.title("Attendance Rate vs Final Score")
    plt.xlabel("Attendance Rate (%)")
    plt.ylabel("Final Score")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "05_attendance_vs_score.png", dpi=150)
    plt.close()

    print(f"\nSaved 5 charts to {OUT_DIR}")


if __name__ == "__main__":
    data = load_clean_data()
    print_summary(data)
    make_plots(data)
