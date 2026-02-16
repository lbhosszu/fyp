import os
import joblib
import numpy as np
import pandas as pd

FEATURE_COLS = [
    "TeamName", "EventName",
    "Year",
    "GridPos", "QualiPos",
    "career_race_count", "is_rookie",
    "drv_avg_finish_w", "drv_avg_quali_w",
    "drv_top10_rate_w", "drv_dnf_rate_w",
    "team_avg_finish_w", "team_avg_quali_w",
    "team_top10_rate_w", "team_dnf_rate_w",
]

MODE_TO_TARGET = {
    "easy": ("Top3", 3, "models/rf_top3.joblib"),
    "medium": ("Top5", 5, "models/rf_top5.joblib"),
    "hard": ("Top10", 10, "models/rf_top10.joblib"),
}

def load_dataset(path="data/dataset_with_features.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing dataset at {path}")
    return pd.read_csv(path)

def load_model(mode: str):
    mode = mode.lower().strip()
    if mode not in MODE_TO_TARGET:
        raise ValueError("mode must be one of: easy, medium, hard")
    _, _, model_path = MODE_TO_TARGET[mode]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model at {model_path}. Run train_rf.py first.")
    return joblib.load(model_path)

def predict_race(df: pd.DataFrame, year: int, event_name: str, mode: str) -> pd.DataFrame:
    mode = mode.lower().strip()
    target, k, _ = MODE_TO_TARGET[mode]
    model = load_model(mode)

    race_df = df[(df["Year"] == year) & (df["EventName"] == event_name)].copy()
    if race_df.empty:
        raise ValueError(f"No rows for {year} {event_name} in dataset.")

    probs = model.predict_proba(race_df[FEATURE_COLS])[:, 1]
    race_df["Prob"] = probs

    out = race_df.sort_values("Prob", ascending=False).copy()
    out["PredictedRank"] = np.arange(1, len(out) + 1)
    out["PredictedTopK"] = (out["PredictedRank"] <= k).astype(int)

    # Useful columns for UI
    return out[[
        "Year", "EventName",
        "Abbreviation", "TeamName",
        "GridPos", "QualiPos",
        "Prob", "PredictedRank", "PredictedTopK",
        "FinishPos"
    ]].reset_index(drop=True), target, k
