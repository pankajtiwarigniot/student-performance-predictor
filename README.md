# 🎓 Student Performance Predictor — Working Version

A complete Python + Machine Learning project that:
- generates a student-performance dataset,
- performs exploratory data analysis,
- trains a Random Forest score regressor,
- trains a Random Forest grade classifier,
- saves trained models,
- provides a simple Streamlit web interface for live predictions.

## 1. Install Python

Use Python 3.10 or newer.

## 2. Open the project folder

```bash
cd student-performance-predictor
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Build the dataset and train the models

```bash
python main.py
```

This creates/updates:
- `data/student_performance.csv`
- `outputs/*.png`
- `models/score_regressor.pkl`
- `models/grade_classifier.pkl`

## 5. Start the working web application

```bash
streamlit run app.py
```

A browser window should open automatically. If it does not, copy the local URL printed by Streamlit into your browser.

## Project structure

```text
student-performance-predictor/
├── app.py
├── main.py
├── requirements.txt
├── data/
│   └── student_performance.csv
├── models/
│   ├── score_regressor.pkl
│   └── grade_classifier.pkl
├── outputs/
│   ├── 01_score_distribution.png
│   ├── 02_study_hours_vs_score.png
│   ├── 03_correlation_heatmap.png
│   ├── 04_grade_by_support.png
│   ├── 05_attendance_vs_score.png
│   ├── 06_predicted_vs_actual.png
│   ├── 07_feature_importance.png
│   └── 08_confusion_matrix.png
└── src/
    ├── __init__.py
    ├── generate_data.py
    ├── eda.py
    └── train_model.py
```

## If the app says models are missing

Run:

```bash
python main.py
```

and then:

```bash
streamlit run app.py
```

## Important

The model is intended for an academic/demo project. Its predictions are estimates and should not be used as official academic decisions.
