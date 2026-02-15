import time
import fastf1
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# Use an absolute cache path that is definitely writable (change if you want)
fastf1.Cache.enable_cache("fastf1_cache")


# -----------------------------
# Utilities
# -----------------------------

def _safe_int(x):
    try:
        return int(x)
    except Exception:
        return np.nan


def get_schedule_with_retries(year: int, retries: int = 5, sleep_s: int = 3) -> pd.DataFrame:
    """
    FastF1 schedule fetching can fail due to transient network/backend issues.
    This wrapper retries with a small backoff.
    """
    last_err = None
    for i in range(retries):
        try:
            return fastf1.get_event_schedule(year)
        except Exception as e:
            last_err = e
            time.sleep(sleep_s * (i + 1))
    raise ValueError(f"Failed to load schedule for {year} after {retries} retries. Last error: {last_err}")


def get_event_date(year: int, event_name: str) -> pd.Timestamp:
    schedule = get_schedule_with_retries(year)
    row = schedule[schedule["EventName"] == event_name]
    if row.empty:
        raise ValueError(f"EventName '{event_name}' not found for year {year}")
    return pd.to_datetime(row.iloc[0]["EventDate"])


def list_events(year: int) -> list[str]:
    schedule = get_schedule_with_retries(year)
    return schedule["EventName"].tolist()


# -----------------------------
# Load one race (and qualifying) into a single per-driver table
# Only loads what we need (no telemetry/weather/messages) to reduce failures & speed up.
# -----------------------------

def load_race_quali_rows(year: int, event_name: str) -> pd.DataFrame:
    race = fastf1.get_session(year, event_name, "R")
    race.load(telemetry=False, weather=False, messages=False)

    quali = fastf1.get_session(year, event_name, "Q")
    quali.load(telemetry=False, weather=False, messages=False)

    rr = race.results.copy()
    qr = quali.results.copy()

    rr["DriverNumber"] = rr["DriverNumber"].astype(str)
    qr["DriverNumber"] = qr["DriverNumber"].astype(str)

    merged = pd.merge(
        rr,
        qr[["DriverNumber", "Position", "TeamName"]],
        on="DriverNumber",
        how="left",
        suffixes=("_race", "_quali")
    )

    merged = merged.rename(columns={
        "Position_race": "FinishPos",
        "Position_quali": "QualiPos",
        "GridPosition": "GridPos",
        "TeamName_race": "TeamName"
    })

    merged["Year"] = int(year)
    merged["EventName"] = event_name

    merged["FinishPos"] = merged["FinishPos"].apply(_safe_int)
    merged["QualiPos"] = merged["QualiPos"].apply(_safe_int)
    merged["GridPos"] = merged["GridPos"].apply(_safe_int)

    # Basic finished flag (varies across seasons, but "Finished" is common)
    merged["DidFinish"] = (merged["Status"] == "Finished").astype(int)

    merged["Top3"] = (merged["FinishPos"] <= 3).astype(int)
    merged["Top5"] = (merged["FinishPos"] <= 5).astype(int)
    merged["Top10"] = (merged["FinishPos"] <= 10).astype(int)

    keep = [
        "Year", "EventName",
        "DriverNumber", "Abbreviation",
        "TeamName", "GridPos", "QualiPos",
        "FinishPos", "DidFinish",
        "Top3", "Top5", "Top10"
    ]
    out = merged[keep].copy()
    out = out.dropna(subset=["GridPos"])

    return out


# -----------------------------
# Build historical dataset
# -----------------------------

def build_history_rows(year_start: int, year_end_inclusive: int) -> pd.DataFrame:
    rows = []
    for y in range(year_start, year_end_inclusive + 1):
        events = list_events(y)
        for ev in events:
            try:
                df = load_race_quali_rows(y, ev)
                df["EventDate"] = get_event_date(y, ev)
                rows.append(df)
                print(f"Loaded {y} - {ev} ({len(df)} drivers)")
            except Exception as e:
                print(f"Skipping {y} {ev} due to error: {e}")

    if not rows:
        raise RuntimeError("No races loaded. Check cache/network and FastF1 setup.")

    return pd.concat(rows, ignore_index=True)


def add_rolling_features(df_all: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    df = df_all.sort_values(["EventDate", "Year", "EventName"]).copy()

    # Rookie + career count (DriverNumber used for rollups only)
    df["career_race_count"] = df.groupby("DriverNumber").cumcount()
    df["is_rookie"] = (df["career_race_count"] == 0).astype(int)

    # Driver rolling
    gdrv = df.groupby("DriverNumber", group_keys=False)
    df["drv_avg_finish_w"] = gdrv["FinishPos"].apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    df["drv_avg_quali_w"] = gdrv["QualiPos"].apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    df["drv_top10_rate_w"] = gdrv["Top10"].apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    df["drv_dnf_rate_w"] = gdrv["DidFinish"].apply(lambda s: 1.0 - s.shift(1).rolling(window, min_periods=1).mean())

    # Team rolling
    gteam = df.groupby("TeamName", group_keys=False)
    df["team_avg_finish_w"] = gteam["FinishPos"].apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    df["team_avg_quali_w"] = gteam["QualiPos"].apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    df["team_top10_rate_w"] = gteam["Top10"].apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    df["team_dnf_rate_w"] = gteam["DidFinish"].apply(lambda s: 1.0 - s.shift(1).rolling(window, min_periods=1).mean())

    # Fill missing driver rollups with team rollups then global mean
    for col_drv, col_team in [
        ("drv_avg_finish_w", "team_avg_finish_w"),
        ("drv_avg_quali_w", "team_avg_quali_w"),
        ("drv_top10_rate_w", "team_top10_rate_w"),
        ("drv_dnf_rate_w", "team_dnf_rate_w"),
    ]:
        df[col_drv] = df[col_drv].fillna(df[col_team])
        df[col_drv] = df[col_drv].fillna(df[col_drv].mean())

    # Fill missing team rollups with global mean
    for col in ["team_avg_finish_w", "team_avg_quali_w", "team_top10_rate_w", "team_dnf_rate_w"]:
        df[col] = df[col].fillna(df[col].mean())

    # Fill missing quali with grid
    df["QualiPos"] = df["QualiPos"].fillna(df["GridPos"])

    return df


# -----------------------------
# Model training (fixed split)
# Year is numeric (important for predicting 2024+ when training ends at 2023)
# -----------------------------

FEATURE_COLS = [
    "TeamName", "EventName",   # categorical
    "Year",                    # numeric
    "GridPos", "QualiPos",
    "career_race_count", "is_rookie",
    "drv_avg_finish_w", "drv_avg_quali_w",
    "drv_top10_rate_w", "drv_dnf_rate_w",
    "team_avg_finish_w", "team_avg_quali_w",
    "team_top10_rate_w", "team_dnf_rate_w",
]

CAT_FEATURES = ["TeamName", "EventName"]
NUM_FEATURES = [c for c in FEATURE_COLS if c not in CAT_FEATURES]


def make_model_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
            ("num", "passthrough", NUM_FEATURES),
        ]
    )

    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=14,
        min_samples_split=6,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    return Pipeline([("prep", pre), ("rf", rf)])


def train_models_fixed_split(df_all_with_roll: pd.DataFrame, train_end_year: int = 2023) -> dict:
    train_df = df_all_with_roll[df_all_with_roll["Year"] <= train_end_year].copy()

    X = train_df[FEATURE_COLS].copy()
    models = {}

    for target in ["Top3", "Top5", "Top10"]:
        pipe = make_model_pipeline()
        y = train_df[target].astype(int).values
        pipe.fit(X, y)
        models[target] = pipe

    return models


def predict_race_fixed_split(
    df_all_with_roll: pd.DataFrame,
    models: dict,
    year: int,
    event_name: str,
    mode: str = "easy"
) -> pd.DataFrame:
    mode = mode.lower().strip()
    if mode not in ["easy", "medium", "hard"]:
        raise ValueError("mode must be one of: easy, medium, hard")

    target_map = {"easy": "Top3", "medium": "Top5", "hard": "Top10"}
    k_map = {"easy": 3, "medium": 5, "hard": 10}
    target = target_map[mode]
    k = k_map[mode]

    race_df = df_all_with_roll[
        (df_all_with_roll["Year"] == year) &
        (df_all_with_roll["EventName"] == event_name)
    ].copy()

    if race_df.empty:
        raise ValueError(f"No rows found for {year} {event_name}. Did it load correctly?")

    X_pred = race_df[FEATURE_COLS].copy()

    probs = models[target].predict_proba(X_pred)[:, 1]
    race_df[f"P_{target}"] = probs

    out = race_df.sort_values(f"P_{target}", ascending=False).copy()
    out["PredictedRank"] = np.arange(1, len(out) + 1)
    out[f"Predicted_{target}"] = (out["PredictedRank"] <= k).astype(int)

    return out[[
        "Year", "EventName",
        "Abbreviation", "TeamName",
        "GridPos", "QualiPos",
        "career_race_count", "is_rookie",
        f"P_{target}", "PredictedRank", f"Predicted_{target}",
        "FinishPos"
    ]].reset_index(drop=True)


# -----------------------------
# Example usage
# -----------------------------

if __name__ == "__main__":
    # Load 2019–2024 (includes your test year)
    df_all = build_history_rows(2019, 2024)

    # Add rolling features (computed using past races only)
    df_all = add_rolling_features(df_all, window=5)

    # Train fixed split models: train on 2019–2023
    models = train_models_fixed_split(df_all, train_end_year=2023)

    # Predict a 2024 race
    year = 2024
    event_name = "Bahrain Grand Prix"  # must match FastF1 schedule EventName exactly
    preds_easy = predict_race_fixed_split(df_all, models, year, event_name, mode="easy")

    print(preds_easy.head(12))
