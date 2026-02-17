import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from fastf1_utils.fastf1_service import get_race_summary


st.set_page_config(page_title="F1 Prediction Game", layout="wide")
st.title("F1 Prediction Game")

DATASET_PATH = "data/dataset_with_features.csv"
MODEL_PATHS = {
    "easy": "models/rf_top3.joblib",
    "medium": "models/rf_top5.joblib",
    "hard": "models/rf_top10.joblib",
}
MODE_TO_K = {"easy": 3, "medium": 5, "hard": 10}

FEATURE_COLS = [
    "TeamName",
    "EventName",
    "Year",
    "GridPos",
    "QualiPos",
    "career_race_count",
    "is_rookie",
    "drv_avg_finish_w",
    "drv_avg_quali_w",
    "drv_top10_rate_w",
    "drv_dnf_rate_w",
    "team_avg_finish_w",
    "team_avg_quali_w",
    "team_top10_rate_w",
    "team_dnf_rate_w",
]


@st.cache_data(show_spinner=False)
def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing dataset: {path}. "
            "Make sure you have data/dataset_with_features.csv"
        )
    df = pd.read_csv(path)

    required = set(["Year", "EventName", "Abbreviation", "TeamName", "FinishPos"] + FEATURE_COLS)
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    return df


@st.cache_resource(show_spinner=False)
def load_model(mode: str):
    model_path = MODEL_PATHS.get(mode)
    if not model_path:
        raise ValueError("mode must be one of: easy, medium, hard")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Missing model: {model_path}. "
            "Run train_rf.py to generate models/"
        )
    return joblib.load(model_path)


def predict_race(df: pd.DataFrame, year: int, event_name: str, mode: str) -> pd.DataFrame:
    k = MODE_TO_K[mode]
    model = load_model(mode)

    race_df = df[(df["Year"] == year) & (df["EventName"] == event_name)].copy()
    if race_df.empty:
        raise ValueError(f"No rows found for {year} {event_name} in dataset.")

    X = race_df[FEATURE_COLS].copy()
    race_df["Prob"] = model.predict_proba(X)[:, 1]

    out = race_df.sort_values("Prob", ascending=False).copy()
    out["PredictedRank"] = np.arange(1, len(out) + 1)
    out["PredictedTopK"] = (out["PredictedRank"] <= k).astype(int)
    out["TrueTopK"] = (out["FinishPos"] <= k).astype(int)

    return out[
        [
            "Abbreviation",
            "TeamName",
            "GridPos",
            "QualiPos",
            "Prob",
            "PredictedRank",
            "PredictedTopK",
            "FinishPos",
            "TrueTopK",
        ]
    ].reset_index(drop=True)


if "race_loaded" not in st.session_state:
    st.session_state.race_loaded = False
if "loaded_key" not in st.session_state:
    st.session_state.loaded_key = None

df = load_dataset(DATASET_PATH)

st.sidebar.header("Race Selection")
years = sorted(df["Year"].unique().tolist())
season = st.sidebar.selectbox("Season", years, index=len(years) - 1)
event_names = sorted(df[df["Year"] == season]["EventName"].unique().tolist())
gp_name = st.sidebar.selectbox("Grand Prix", event_names, index=0)
mode = st.sidebar.radio("Game mode", ["easy", "medium", "hard"], horizontal=True)

selected_key = f"{season}|{gp_name}"
if st.session_state.loaded_key != selected_key:
    st.session_state.race_loaded = False

if st.sidebar.button("Load race data"):
    try:
        with st.spinner("Loading race data..."):
            results, weather = get_race_summary(season, gp_name)
        st.session_state.race_loaded = True
        st.session_state.loaded_key = selected_key
        st.session_state.race_results = results
        st.session_state.weather = weather
    except Exception as exc:
        st.session_state.race_loaded = False
        st.error(f"Failed to load race data: {exc}")

if st.session_state.race_loaded and st.session_state.loaded_key == selected_key:
    st.subheader(f"{gp_name} {season} race summary")
    st.dataframe(st.session_state.race_results, use_container_width=True)

    st.subheader("Weather summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg air temp (°C)", f"{st.session_state.weather['avg_air_temp']:.1f}")
    c2.metric("Avg track temp (°C)", f"{st.session_state.weather['avg_track_temp']:.1f}")
    c3.metric("Max wind (m/s)", f"{st.session_state.weather['max_wind']:.1f}")
    c4.metric("Rain", "Yes" if st.session_state.weather["rain_any"] else "No")

    st.markdown("---")
    st.subheader("Play the prediction game")
    k = MODE_TO_K[mode]
    if st.button(f"Predict Top{k}"):
        with st.spinner("Running RF prediction..."):
            preds = predict_race(df, season, gp_name, mode)

        score = int(preds.head(k)["TrueTopK"].sum())
        st.success(f"Score: {score}/{k} correct drivers in the Top{k}")
        st.dataframe(preds, use_container_width=True)
else:
    st.info("Select a season and GP, then click 'Load race data' to unlock the prediction game.")
