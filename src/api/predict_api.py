"""
predict_api.py
==============
Standalone prediction helper module.

Provides a clean API for loading the trained RF models and generating
race predictions. This was the original prediction module before the
prediction logic was also embedded directly in app.py.

Maps game difficulty modes to the appropriate model:
  easy   → rf_top3.joblib  (predict podium)
  medium → rf_top5.joblib  (predict top 5)
  hard   → rf_top10.joblib (predict top 10)
"""

import os
import joblib
import numpy as np
import pandas as pd

# Feature columns — must match what the model was trained on in train_models.py
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

# Maps difficulty mode to (target column, K value, model file path)
MODE_TO_TARGET = {
    "easy": ("Top3", 3, "models/rf_top3.joblib"),
    "medium": ("Top5", 5, "models/rf_top5.joblib"),
    "hard": ("Top10", 10, "models/rf_top10.joblib"),
}


def load_dataset(path="data/dataset_with_features.csv") -> pd.DataFrame:
    """Load the feature-engineered dataset from CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing dataset at {path}")
    return pd.read_csv(path)

def load_model(mode: str):
    """Load the trained RF model for the given difficulty mode."""
    mode = mode.lower().strip()
    if mode not in MODE_TO_TARGET:
        raise ValueError("mode must be one of: easy, medium, hard")
    _, _, model_path = MODE_TO_TARGET[mode]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model at {model_path}. Run train_rf.py first.")
    return joblib.load(model_path)

def predict_race(df: pd.DataFrame, year: int, event_name: str, mode: str) -> pd.DataFrame:
    """Generate ranked predictions for a specific race.

    Uses predict_proba to get each driver's probability of finishing in
    the top K, then ranks all drivers by that probability. Returns a
    DataFrame sorted by predicted rank (best prediction first).

    Args:
        df: Full feature-engineered dataset
        year: Season year
        event_name: Grand Prix name
        mode: Difficulty ("easy", "medium", "hard")

    Returns:
        Tuple of (predictions DataFrame, target name, K value)
    """
    mode = mode.lower().strip()
    target, k, _ = MODE_TO_TARGET[mode]
    model = load_model(mode)

    # Filter to just this race's rows
    race_df = df[(df["Year"] == year) & (df["EventName"] == event_name)].copy()
    if race_df.empty:
        raise ValueError(f"No rows for {year} {event_name} in dataset.")

    # predict_proba[:, 1] = probability of the positive class (in top K)
    probs = model.predict_proba(race_df[FEATURE_COLS])[:, 1]
    race_df["Prob"] = probs

    # Sort by probability descending — highest prob = model's P1 pick
    out = race_df.sort_values("Prob", ascending=False).copy()
    out["PredictedRank"] = np.arange(1, len(out) + 1)
    out["PredictedTopK"] = (out["PredictedRank"] <= k).astype(int)

    # Return only the columns needed by the UI
    return out[[
        "Year", "EventName",
        "Abbreviation", "TeamName",
        "GridPos", "QualiPos",
        "Prob", "PredictedRank", "PredictedTopK",
        "FinishPos"
    ]].reset_index(drop=True), target, k
