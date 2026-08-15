@echo off
echo Installing required packages...
python -m pip install -r requirements.txt
echo.
echo Training models...
python main.py
echo.
echo Starting Student Performance Predictor...
python -m streamlit run app.py
pause
