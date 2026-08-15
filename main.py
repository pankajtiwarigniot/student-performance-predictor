from pathlib import Path
import joblib

from src.generate_data import generate_dataset, OUTPUT_PATH
from src.eda import load_clean_data, print_summary, make_plots
from src.train_model import (
    load_data,
    train_regressor,
    train_classifier,
    MODEL_DIR,
    OUT_DIR,
)

BASE_DIR = Path(__file__).resolve().parent


def main():
    print("STEP 1/3 - Generating dataset...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate_dataset()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Created {len(df)} student records: {OUTPUT_PATH}")

    print("\nSTEP 2/3 - Running EDA...")
    clean_df = load_clean_data()
    print_summary(clean_df)
    make_plots(clean_df)

    print("\nSTEP 3/3 - Training models...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data()
    reg_pipeline = train_regressor(data)
    clf_pipeline = train_classifier(data)

    joblib.dump(reg_pipeline, MODEL_DIR / "score_regressor.pkl")
    joblib.dump(clf_pipeline, MODEL_DIR / "grade_classifier.pkl")

    print("\nSUCCESS!")
    print(f"Models: {MODEL_DIR}")
    print(f"Charts: {OUT_DIR}")
    print("\nTo launch the web app:")
    print("  streamlit run app.py")


if __name__ == "__main__":
    main()
