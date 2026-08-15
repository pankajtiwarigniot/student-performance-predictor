"""
generate_data.py
-----------------
Generates a synthetic-but-realistic student performance dataset.

Why synthetic data?
Public "student performance" datasets (e.g. UCI) exist, but generating our
own lets the project run fully offline, keeps the repo self-contained, and
lets us control the underlying relationships (so we can sanity-check the
model later). The generator injects real-world-like noise and correlations
(e.g. study hours + attendance push scores up, sleep deprivation and high
absences pull them down) so the ML model has genuine signal to learn.

Run:
    python src/generate_data.py
Outputs:
    data/student_performance.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42
N_STUDENTS = 1200

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "student_performance.csv"


def generate_dataset(n_students: int = N_STUDENTS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    student_id = np.arange(1, n_students + 1)
    gender = rng.choice(["Male", "Female"], size=n_students, p=[0.52, 0.48])

    study_hours_per_week = np.clip(rng.normal(15, 6, n_students), 0, 40)
    attendance_rate = np.clip(rng.normal(85, 10, n_students), 40, 100)
    sleep_hours = np.clip(rng.normal(6.8, 1.3, n_students), 3, 10)
    previous_gpa = np.clip(rng.normal(2.9, 0.6, n_students), 0.0, 4.0)
    extracurricular_hours = np.clip(rng.exponential(3, n_students), 0, 20)
    parental_support = rng.choice(["Low", "Medium", "High"], size=n_students, p=[0.25, 0.45, 0.30])
    internet_access = rng.choice(["Yes", "No"], size=n_students, p=[0.85, 0.15])
    study_group = rng.choice(["Yes", "No"], size=n_students, p=[0.4, 0.6])

    support_bonus = pd.Series(parental_support).map({"Low": -3, "Medium": 0, "High": 3}).to_numpy()
    internet_bonus = pd.Series(internet_access).map({"Yes": 2, "No": -2}).to_numpy()
    group_bonus = pd.Series(study_group).map({"Yes": 1.5, "No": 0}).to_numpy()

    # Underlying "true" score model (what we want the ML model to rediscover)
    final_score = (
        18
        + 0.55 * study_hours_per_week
        + 0.22 * attendance_rate
        + 6.0 * previous_gpa
        + 0.8 * sleep_hours
        + 0.25 * extracurricular_hours
        + support_bonus
        + internet_bonus
        + group_bonus
        + rng.normal(0, 4.5, n_students)  # noise
    )
    final_score = np.clip(final_score, 0, 100)

    df = pd.DataFrame({
        "student_id": student_id,
        "gender": gender,
        "study_hours_per_week": study_hours_per_week.round(1),
        "attendance_rate": attendance_rate.round(1),
        "sleep_hours": sleep_hours.round(1),
        "previous_gpa": previous_gpa.round(2),
        "extracurricular_hours": extracurricular_hours.round(1),
        "parental_support": parental_support,
        "internet_access": internet_access,
        "study_group": study_group,
        "final_score": final_score.round(1),
    })

    # Derived label used for the classification task
    def grade_bucket(score: float) -> str:
        if score >= 85:
            return "A"
        if score >= 70:
            return "B"
        if score >= 55:
            return "C"
        if score >= 40:
            return "D"
        return "F"

    df["grade"] = df["final_score"].apply(grade_bucket)

    # Sprinkle a few missing values to make cleaning realistic
    missing_idx = rng.choice(df.index, size=int(0.02 * n_students), replace=False)
    df.loc[missing_idx, "sleep_hours"] = np.nan

    return df


if __name__ == "__main__":
    dataset = generate_dataset()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT_PATH, index=False)
    print(f"Generated {len(dataset)} rows -> {OUTPUT_PATH}")
    print(dataset.head())
