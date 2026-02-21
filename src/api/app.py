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


def get_race_driver_pool(df: pd.DataFrame, year: int, event_name: str) -> pd.DataFrame:
    race_df = df[(df["Year"] == year) & (df["EventName"] == event_name)].copy()
    race_df = race_df[["Abbreviation", "TeamName", "GridPos", "FinishPos"]].drop_duplicates(
        subset=["Abbreviation"]
    )
    return race_df.sort_values("GridPos").reset_index(drop=True)


if "race_loaded" not in st.session_state:
    st.session_state.race_loaded = False
if "loaded_key" not in st.session_state:
    st.session_state.loaded_key = None
if "game_key" not in st.session_state:
    st.session_state.game_key = None
if "game_submitted" not in st.session_state:
    st.session_state.game_submitted = False
if "game_result" not in st.session_state:
    st.session_state.game_result = None
if "pick_nonce" not in st.session_state:
    st.session_state.pick_nonce = 0

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

current_game_key = f"{selected_key}|{mode}"
if st.session_state.game_key != current_game_key:
    st.session_state.game_key = current_game_key
    st.session_state.game_submitted = False
    st.session_state.game_result = None
    st.session_state.pick_nonce += 1

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
    race_pool = get_race_driver_pool(df, season, gp_name)
    if race_pool.empty:
        st.warning("No race entries found in the dataset for this event.")
    else:
        driver_labels = [f"{row.Abbreviation} - {row.TeamName}" for row in race_pool.itertuples()]
        label_to_abbr = {
            f"{row.Abbreviation} - {row.TeamName}": row.Abbreviation for row in race_pool.itertuples()
        }
        abbr_to_team = dict(zip(race_pool["Abbreviation"], race_pool["TeamName"]))
        form_id = f"{current_game_key}_{st.session_state.pick_nonce}"

        st.caption(f"Pick your predicted finishing order for Top{k}.")
        with st.form(f"prediction_form_{form_id}"):
            picks = []
            for pos in range(1, k + 1):
                pick_label = st.selectbox(
                    f"P{pos}",
                    options=["Select driver"] + driver_labels,
                    index=0,
                    key=f"user_pick_{form_id}_{pos}",
                )
                picks.append(pick_label)
            submitted = st.form_submit_button("Submit prediction")

        if submitted:
            if any(p == "Select driver" for p in picks):
                st.error("Please select a driver for every position before submitting.")
            elif len(set(picks)) != len(picks):
                st.error("Each position must have a different driver.")
            else:
                user_pick_abbr = [label_to_abbr[p] for p in picks]
                actual_topk_df = race_pool.sort_values("FinishPos").head(k)
                actual_topk_abbr = actual_topk_df["Abbreviation"].tolist()
                model_preds = predict_race(df, season, gp_name, mode)
                model_topk_abbr = model_preds.head(k)["Abbreviation"].tolist()

                exact_hits = sum(u == a for u, a in zip(user_pick_abbr, actual_topk_abbr))
                in_topk_hits = sum(u in set(actual_topk_abbr) for u in user_pick_abbr)
                points = exact_hits * 3 + (in_topk_hits - exact_hits)
                model_exact_hits = sum(m == a for m, a in zip(model_topk_abbr, actual_topk_abbr))
                model_in_topk_hits = sum(m in set(actual_topk_abbr) for m in model_topk_abbr)
                model_points = model_exact_hits * 3 + (model_in_topk_hits - model_exact_hits)

                st.session_state.game_result = {
                    "k": k,
                    "user_pick_abbr": user_pick_abbr,
                    "actual_topk_abbr": actual_topk_abbr,
                    "model_topk_abbr": model_topk_abbr,
                    "exact_hits": exact_hits,
                    "in_topk_hits": in_topk_hits,
                    "points": points,
                    "model_exact_hits": model_exact_hits,
                    "model_in_topk_hits": model_in_topk_hits,
                    "model_points": model_points,
                    "model_preds": model_preds,
                    "abbr_to_team": abbr_to_team,
                }
                st.session_state.game_submitted = True

        if st.session_state.game_submitted and st.session_state.game_result:
            result = st.session_state.game_result
            actual_set = set(result["actual_topk_abbr"])

            st.markdown("### Your results")
            u1, u2, u3 = st.columns(3)
            u1.metric("Exact position hits", f"{result['exact_hits']}/{result['k']}")
            u2.metric("Drivers in Top-K", f"{result['in_topk_hits']}/{result['k']}")
            u3.metric("Points", f"{result['points']}/{result['k'] * 3}")

            user_rows = []
            for i in range(result["k"]):
                user_abbr = result["user_pick_abbr"][i]
                actual_abbr = result["actual_topk_abbr"][i]
                user_rows.append(
                    {
                        "Position": i + 1,
                        "Your pick": f"{user_abbr} ({result['abbr_to_team'].get(user_abbr, '')})",
                        "Actual": f"{actual_abbr} ({result['abbr_to_team'].get(actual_abbr, '')})",
                        "Exact hit": user_abbr == actual_abbr,
                        "In actual Top-K": user_abbr in actual_set,
                    }
                )
            st.dataframe(pd.DataFrame(user_rows), use_container_width=True)

            st.markdown("### Model results")
            m1, m2, m3 = st.columns(3)
            m1.metric("Exact position hits", f"{result['model_exact_hits']}/{result['k']}")
            m2.metric("Drivers in Top-K", f"{result['model_in_topk_hits']}/{result['k']}")
            m3.metric("Points", f"{result['model_points']}/{result['k'] * 3}")

            model_rows = []
            for i in range(result["k"]):
                model_abbr = result["model_topk_abbr"][i]
                actual_abbr = result["actual_topk_abbr"][i]
                model_rows.append(
                    {
                        "Position": i + 1,
                        "Model pick": f"{model_abbr} ({result['abbr_to_team'].get(model_abbr, '')})",
                        "Actual": f"{actual_abbr} ({result['abbr_to_team'].get(actual_abbr, '')})",
                        "Exact hit": model_abbr == actual_abbr,
                        "In actual Top-K": model_abbr in actual_set,
                    }
                )
            st.dataframe(pd.DataFrame(model_rows), use_container_width=True)

            with st.expander("See full model prediction table"):
                st.dataframe(result["model_preds"], use_container_width=True)

            if st.button("Play again"):
                st.session_state.game_submitted = False
                st.session_state.game_result = None
                st.session_state.pick_nonce += 1
                st.rerun()
else:
    st.info("Select a season and GP, then click 'Load race data' to unlock the prediction game.")
