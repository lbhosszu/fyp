import os
import re
import unicodedata
import base64

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from fastf1_utils.fastf1_service import get_race_summary, get_qualifying_results, get_circuit_info


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


def get_driver_form(df: pd.DataFrame, year: int, event_name: str) -> pd.DataFrame:
    """Get the rolling driver & team form stats for the selected race.
    These are the same features the model sees at prediction time."""
    race_df = df[(df["Year"] == year) & (df["EventName"] == event_name)].copy()
    if race_df.empty:
        return pd.DataFrame()

    form = race_df[[
        "Abbreviation", "TeamName", "GridPos",
        "career_race_count", "is_rookie",
        "drv_avg_finish_w",
        "drv_top10_rate_w", "drv_dnf_rate_w",
        "team_avg_finish_w",
        "team_top10_rate_w",
    ]].copy()

    form = form.rename(columns={
        "Abbreviation": "Driver",
        "TeamName": "Team",
        "GridPos": "Grid",
        "career_race_count": "Races",
        "is_rookie": "Rookie",
        "drv_avg_finish_w": "Avg Finish (L5)",
        "drv_top10_rate_w": "Top-10 Rate (L5)",
        "drv_dnf_rate_w": "DNF Rate (L5)",
        "team_avg_finish_w": "Team Avg Finish (L5)",
        "team_top10_rate_w": "Team Top-10 Rate (L5)",
    })

    # Round floats for readability
    float_cols = [
        "Avg Finish (L5)", "Top-10 Rate (L5)",
        "DNF Rate (L5)", "Team Avg Finish (L5)",
        "Team Top-10 Rate (L5)",
    ]
    for col in float_cols:
        form[col] = form[col].round(2)

    form["Rookie"] = form["Rookie"].map({1: "Yes", 0: ""})
    form = form.sort_values("Grid").reset_index(drop=True)
    return form


def _normalize_name_key(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def find_track_layout_path(year: int, event_name: str) -> str | None:
    year_dir = os.path.join("src", "api", "track_layouts", str(year))
    if not os.path.isdir(year_dir):
        return None

    target_key = _normalize_name_key(event_name)
    for filename in os.listdir(year_dir):
        if filename.lower().endswith(".png") and _normalize_name_key(os.path.splitext(filename)[0]) == target_key:
            return os.path.join(year_dir, filename)
    return None


def show_responsive_track_image(image_path: str) -> None:
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    st.markdown(
        f"""
        <div style="display:flex;justify-content:center;">
          <div style="width:min(100%, 560px);height:clamp(180px, 34vh, 320px);">
            <img
              src="data:image/png;base64,{encoded}"
              style="width:100%;height:100%;object-fit:contain;image-rendering:auto;"
            />
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if "ui_season" not in st.session_state:
    st.session_state.ui_season = None
if "ui_race_index" not in st.session_state:
    st.session_state.ui_race_index = 0
if "track_select" not in st.session_state:
    st.session_state.track_select = None
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
years = sorted(df["Year"].unique().tolist())
season = st.selectbox("Season", years, index=len(years) - 1)

if st.session_state.ui_season != season:
    st.session_state.ui_season = season
    st.session_state.ui_race_index = 0
    st.session_state.race_loaded = False
    st.session_state.loaded_key = None
    st.session_state.game_submitted = False
    st.session_state.game_result = None
    st.session_state.pick_nonce += 1
    st.session_state.track_select = None

event_names = sorted(df[df["Year"] == season]["EventName"].unique().tolist())
if not event_names:
    st.error("No races found for this season in the dataset.")
    st.stop()

st.session_state.ui_race_index = st.session_state.ui_race_index % len(event_names)
current_index = st.session_state.ui_race_index
if st.session_state.track_select is None or st.session_state.track_select not in event_names:
    st.session_state.track_select = event_names[current_index]

left_col, center_col, right_col = st.columns([1, 8, 1])
with left_col:
    if st.button("◀", use_container_width=True):
        st.session_state.ui_race_index = (st.session_state.ui_race_index - 1) % len(event_names)
        st.session_state.race_loaded = False
        st.session_state.track_select = event_names[st.session_state.ui_race_index]
with right_col:
    if st.button("▶", use_container_width=True):
        st.session_state.ui_race_index = (st.session_state.ui_race_index + 1) % len(event_names)
        st.session_state.race_loaded = False
        st.session_state.track_select = event_names[st.session_state.ui_race_index]

with center_col:
    st.selectbox(
        "Track",
        options=event_names,
        key="track_select",
        label_visibility="collapsed",
    )
    if st.session_state.track_select != event_names[st.session_state.ui_race_index]:
        st.session_state.ui_race_index = event_names.index(st.session_state.track_select)
        st.session_state.race_loaded = False

gp_name = event_names[st.session_state.ui_race_index]
selected_key = f"{season}|{gp_name}"

with center_col:
    st.markdown(f"<h3 style='text-align:center;margin-bottom:0.5rem;'>{gp_name}</h3>", unsafe_allow_html=True)
    image_path = find_track_layout_path(season, gp_name)
    if image_path and os.path.exists(image_path):
        show_responsive_track_image(image_path)
    else:
        st.info(f"No track layout image found for {gp_name} ({season}).")

st.markdown("---")
controls_left, controls_right = st.columns([3, 1])
with controls_left:
    mode = st.radio("Game mode", ["easy", "medium", "hard"], horizontal=True)
with controls_right:
    st.write("")
    st.write("")
    load_clicked = st.button("Load race data", use_container_width=True)

if st.session_state.loaded_key != selected_key:
    st.session_state.race_loaded = False

current_game_key = f"{selected_key}|{mode}"
if st.session_state.game_key != current_game_key:
    st.session_state.game_key = current_game_key
    st.session_state.game_submitted = False
    st.session_state.game_result = None
    st.session_state.pick_nonce += 1

if load_clicked:
    try:
        with st.spinner("Loading race data..."):
            results, weather = get_race_summary(season, gp_name)

            # Qualifying — load separately so a failure doesn't block everything
            try:
                quali_df, name_map = get_qualifying_results(season, gp_name)
            except Exception:
                quali_df = pd.DataFrame()
                name_map = {}

            # Circuit info
            try:
                circuit = get_circuit_info(season, gp_name)
            except Exception:
                circuit = {}

        st.session_state.race_loaded = True
        st.session_state.loaded_key = selected_key
        st.session_state.race_results = results
        st.session_state.weather = weather
        st.session_state.quali_results = quali_df
        st.session_state.driver_names = name_map
        st.session_state.circuit_info = circuit
    except Exception as exc:
        st.session_state.race_loaded = False
        st.error(f"Failed to load race data: {exc}")

if st.session_state.race_loaded and st.session_state.loaded_key == selected_key:

    # ── Circuit Info ──
    circuit = st.session_state.get("circuit_info", {})
    if circuit:
        info_parts = []
        if circuit.get("location") and circuit.get("country"):
            info_parts.append(f"**Location:** {circuit['location']}, {circuit['country']}")
        if circuit.get("event_date"):
            info_parts.append(f"**Date:** {circuit['event_date']}")
        if info_parts:
            st.caption(" · ".join(info_parts))

    # ── Weather Summary ──
    st.subheader(f"{gp_name} {season} — race conditions")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg air temp (°C)", f"{st.session_state.weather['avg_air_temp']:.1f}")
    c2.metric("Avg track temp (°C)", f"{st.session_state.weather['avg_track_temp']:.1f}")
    c3.metric("Max wind (m/s)", f"{st.session_state.weather['max_wind']:.1f}")
    c4.metric("Rain", "Yes" if st.session_state.weather["rain_any"] else "No")

    st.markdown("---")

    # ── Qualifying Results ──
    quali_df = st.session_state.get("quali_results", pd.DataFrame())
    if not quali_df.empty:
        st.subheader("Qualifying results")
        st.caption("Session lap times — empty cells indicate the driver was eliminated in an earlier session.")
        st.dataframe(quali_df, use_container_width=True, hide_index=True)
    st.markdown("---")

    # ── Driver & Team Form Guide ──
    st.subheader("Driver form guide")
    st.caption("Rolling averages over each driver's last 5 races going into this event (L5 = last 5).")
    name_map = st.session_state.get("driver_names", {})
    form_df = get_driver_form(df, season, gp_name)
    if not form_df.empty:
        # Replace abbreviations with full names where available
        form_df["Driver"] = form_df["Driver"].map(lambda abbr: name_map.get(abbr, abbr))
        tab_drv, tab_team = st.tabs(["Driver stats", "Team stats"])
        with tab_drv:
            drv_cols = ["Driver", "Team", "Grid", "Avg Finish (L5)",
                        "Top-10 Rate (L5)", "DNF Rate (L5)",
                        "Races", "Rookie"]
            st.dataframe(
                form_df[drv_cols].style
                    .background_gradient(subset=["Avg Finish (L5)"], cmap="RdYlGn_r")
                    .background_gradient(subset=["Top-10 Rate (L5)"], cmap="RdYlGn")
                    .format(precision=2),
                use_container_width=True,
                hide_index=True,
            )
        with tab_team:
            team_cols = ["Driver", "Team", "Grid", "Team Avg Finish (L5)",
                         "Team Top-10 Rate (L5)"]
            st.dataframe(
                form_df[team_cols].style
                    .background_gradient(subset=["Team Avg Finish (L5)"], cmap="RdYlGn_r")
                    .background_gradient(subset=["Team Top-10 Rate (L5)"], cmap="RdYlGn")
                    .format(precision=2),
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info("No form data available for this race.")

    st.markdown("---")
    st.subheader("Play the prediction game")
    k = MODE_TO_K[mode]
    race_pool = get_race_driver_pool(df, season, gp_name)
    if race_pool.empty:
        st.warning("No race entries found in the dataset for this event.")
    else:
        driver_labels = [
            f"{name_map.get(row.Abbreviation, row.Abbreviation)} - {row.TeamName}"
            for row in race_pool.itertuples()
        ]
        label_to_abbr = {
            f"{name_map.get(row.Abbreviation, row.Abbreviation)} - {row.TeamName}": row.Abbreviation
            for row in race_pool.itertuples()
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

            def _display_name(abbr, team_map):
                full = name_map.get(abbr, abbr)
                team = team_map.get(abbr, "")
                return f"{full} ({team})" if team else full

            user_rows = []
            for i in range(result["k"]):
                user_abbr = result["user_pick_abbr"][i]
                actual_abbr = result["actual_topk_abbr"][i]
                user_rows.append(
                    {
                        "Position": i + 1,
                        "Your pick": _display_name(user_abbr, result["abbr_to_team"]),
                        "Actual": _display_name(actual_abbr, result["abbr_to_team"]),
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
                        "Model pick": _display_name(model_abbr, result["abbr_to_team"]),
                        "Actual": _display_name(actual_abbr, result["abbr_to_team"]),
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
    st.info("Choose a season/race and click 'Load race data' to unlock the game.")
