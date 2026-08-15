import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Student Performance Predictor")
st.write(
    "Enter a student's academic and lifestyle information to predict "
    "their final score and expected grade."
)

REG_MODEL = MODEL_DIR / "score_regressor.pkl"
CLF_MODEL = MODEL_DIR / "grade_classifier.pkl"

@st.cache_resource
def load_models():
    if not REG_MODEL.exists() or not CLF_MODEL.exists():
        return None, None
    return joblib.load(REG_MODEL), joblib.load(CLF_MODEL)

regressor, classifier = load_models()

if regressor is None or classifier is None:
    st.error("Trained models were not found.")
    st.info("Run `python main.py` first, then restart this app.")
    st.stop()

with st.sidebar:
    st.header("Student Details")
    study_hours = st.slider("Study hours / week", 0.0, 40.0, 15.0, 0.5)
    attendance = st.slider("Attendance rate (%)", 40.0, 100.0, 85.0, 0.5)
    sleep = st.slider("Sleep hours / night", 3.0, 10.0, 7.0, 0.5)
    previous_gpa = st.slider("Previous GPA", 0.0, 4.0, 2.9, 0.1)
    extracurricular = st.slider(
        "Extracurricular hours / week", 0.0, 20.0, 3.0, 0.5
    )

    gender = st.selectbox("Gender", ["Male", "Female"])
    parental_support = st.selectbox(
        "Parental support", ["Low", "Medium", "High"]
    )
    internet_access = st.selectbox("Internet access", ["Yes", "No"])
    study_group = st.selectbox("Study group", ["Yes", "No"])

input_df = pd.DataFrame([{
    "study_hours_per_week": study_hours,
    "attendance_rate": attendance,
    "sleep_hours": sleep,
    "previous_gpa": previous_gpa,
    "extracurricular_hours": extracurricular,
    "gender": gender,
    "parental_support": parental_support,
    "internet_access": internet_access,
    "study_group": study_group,
}])

predicted_score = float(regressor.predict(input_df)[0])
predicted_score = max(0.0, min(100.0, predicted_score))
predicted_grade = str(classifier.predict(input_df)[0])

col1, col2 = st.columns(2)
with col1:
    st.metric("Predicted Final Score", f"{predicted_score:.1f} / 100")
with col2:
    st.metric("Predicted Grade", predicted_grade)

st.subheader("Prediction Details")
st.dataframe(input_df, use_container_width=True, hide_index=True)

if predicted_grade in ["A", "B"]:
    st.success("The model predicts a good academic outcome.")
elif predicted_grade == "C":
    st.warning("The model predicts an average outcome. Consistent study may help.")
else:
    st.error("The model predicts that additional academic support may be useful.")

st.caption(
    "This is an educational machine-learning project. Predictions are estimates "
    "and should not be treated as official academic decisions."
)
