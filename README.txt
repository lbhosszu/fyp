F1 Race Prediction Game
=======================
Student:    Laszlo Balazs Hosszu
Student No: 21442296
Supervisor: Charles Markham

Project Description
-------------------
An interactive Formula 1 race-prediction application that uses a Random Forest
classifier trained on seven seasons of F1 data (2018-2024) to predict finishing
positions. Users can play a prediction game against the model through a
Streamlit web interface.

Requirements
------------
Python 3.10 or higher. All dependencies are listed in requirements.txt.

    pip install -r requirements.txt

FastF1 requires an internet connection on first run to download and cache
session data. Subsequent runs use the local cache in fastf1_cache/.

Project Structure
-----------------
src/
  api/
    app.py              - Streamlit web application (main entry point)
    predict_api.py      - Prediction helper used by the app
    track_layouts/      - SVG track layout images
  fastf1_rf_model/
    extract_year.py     - Downloads race/quali data via FastF1
    build_dataset.py    - Feature engineering (rolling windows)
    train_models.py     - Trains LR, RF, and GB classifiers
    evaluate_models.py  - Model comparison and evaluation plots
    ablation_study.py   - Feature ablation analysis
    rolling_cv.py       - Rolling cross-validation across seasons

data/                   - Raw and processed CSV datasets
models/                 - Trained model files (.joblib)
evaluation/             - Evaluation plots (PNG) and result tables (CSV)
report/                 - Final project report (.docx)
fastf1_cache/           - Cached FastF1 session data

How to Run
----------
1. Train models (optional — pre-trained models are included):

       python -m src.fastf1_rf_model.train_models

2. Launch the Streamlit app:

       streamlit run src/api/app.py

3. Open the URL shown in the terminal (usually http://localhost:8501).

How to Play
-----------
1. Select a season and Grand Prix from the sidebar.
2. Choose "Game Mode" to play the prediction game.
3. Pick your predicted Top 5 finishers using the dropdown menus.
4. Press "Submit Predictions" to see your score compared to the model.

Scoring: 3 points for an exact position match, 1 point for a correct driver
in the top 5 but in the wrong position.

Reproducing the Evaluation
--------------------------
To regenerate the evaluation plots and tables:

    python -m src.fastf1_rf_model.evaluate_models
    python -m src.fastf1_rf_model.ablation_study
    python -m src.fastf1_rf_model.rolling_cv

Output files are saved to the evaluation/ directory.
