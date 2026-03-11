"""
build_dataset.py
================
Feature engineering module — the second step in the ML pipeline.

Takes the raw per-year CSVs produced by extract_year.py and computes
rolling performance features that the ML models use for prediction.

Pipeline position:
  extract_year.py  →  build_dataset.py  →  train_models.py  →  evaluate_models.py

Key design decisions:
  - Rolling window of 5 races: captures recent form without being too noisy
  - shift(1) on every rolling feature: prevents DATA LEAKAGE by ensuring
    we only use information available BEFORE the current race
  - Cascading imputation: driver NaN → team mean → global mean
  - Rookie flag covers the driver's entire first season (not just first race)

Output: data/dataset_with_features.csv — the complete dataset ready for training
"""

import os
import pandas as pd
import numpy as np


def add_rolling_features(df_all: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Compute rolling performance features for drivers and teams.

    For each driver/team, calculates statistics over a sliding window of
    their most recent races. The shift(1) call is CRITICAL — it ensures
    each feature only uses data from PREVIOUS races, preventing the model
    from "seeing the future" (data leakage).

    Features computed:
      Driver-level (grouped by DriverNumber):
        - drv_avg_finish_w:  Average finishing position over last 5 races
        - drv_avg_quali_w:   Average qualifying position over last 5 races
        - drv_top10_rate_w:  Proportion of top-10 finishes in last 5 races
        - drv_dnf_rate_w:    Proportion of DNFs in last 5 races

      Team-level (grouped by TeamName):
        - team_avg_finish_w: Team's average finishing position over last 5 races
        - team_avg_quali_w:  Team's average qualifying position over last 5 races
        - team_top10_rate_w: Team's top-10 rate over last 5 races
        - team_dnf_rate_w:   Team's DNF rate over last 5 races

      Career/context features:
        - career_race_count: Cumulative number of races the driver has done
        - is_rookie:         1 if driver is in their first season in the dataset

    Args:
        df_all: Raw dataset with all years concatenated
        window: Number of past races to include in rolling window (default: 5)

    Returns:
        DataFrame with all original columns plus the new rolling features
    """
    # Sort chronologically — essential for rolling calculations to be correct
    df_all["EventDate"] = pd.to_datetime(df_all["EventDate"])
    df = df_all.sort_values(["EventDate", "Year", "EventName"]).copy()

    # Career race count: cumulative count of races per driver (0-indexed)
    df["career_race_count"] = df.groupby("DriverNumber").cumcount()

    # Rookie flag: a driver is a rookie for their ENTIRE first season
    # (not just their first race), matching how F1 defines rookie status
    first_season = df.groupby("DriverNumber")["Year"].transform("min")
    df["is_rookie"] = (df["Year"] == first_season).astype(int)

    # ── Driver-level rolling features ──
    # shift(1) = exclude current race → only use past data (prevents leakage)
    # min_periods=1 = allow computation even with fewer than 5 prior races
    gdrv = df.groupby("DriverNumber", group_keys=False)
    df["drv_avg_finish_w"] = gdrv["FinishPos"].apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    df["drv_avg_quali_w"] = gdrv["QualiPos"].apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    df["drv_top10_rate_w"] = gdrv["Top10"].apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    df["drv_dnf_rate_w"] = gdrv["DidFinish"].apply(lambda s: 1.0 - s.shift(1).rolling(window, min_periods=1).mean())

    # ── Team-level rolling features ──
    # Same logic but grouped by team — captures car performance trends
    gteam = df.groupby("TeamName", group_keys=False)
    df["team_avg_finish_w"] = gteam["FinishPos"].apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    df["team_avg_quali_w"] = gteam["QualiPos"].apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    df["team_top10_rate_w"] = gteam["Top10"].apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    df["team_dnf_rate_w"] = gteam["DidFinish"].apply(lambda s: 1.0 - s.shift(1).rolling(window, min_periods=1).mean())

    # ── Cascading imputation for missing values ──
    # A driver's first race will have NaN rolling features (no history yet).
    # Strategy: fill driver NaN with their team's value, then global mean.
    # This is better than just using 0 — a rookie at Red Bull should get
    # Red Bull's team average, not a meaningless default.
    for col_drv, col_team in [
        ("drv_avg_finish_w", "team_avg_finish_w"),
        ("drv_avg_quali_w", "team_avg_quali_w"),
        ("drv_top10_rate_w", "team_top10_rate_w"),
        ("drv_dnf_rate_w", "team_dnf_rate_w"),
    ]:
        df[col_drv] = df[col_drv].fillna(df[col_team]).fillna(df[col_drv].mean())

    # Fill remaining team-level NaNs with the global mean
    for col in ["team_avg_finish_w", "team_avg_quali_w", "team_top10_rate_w", "team_dnf_rate_w"]:
        df[col] = df[col].fillna(df[col].mean())

    # If qualifying position is missing (e.g. DNS), use grid position as fallback
    df["QualiPos"] = df["QualiPos"].fillna(df["GridPos"])

    return df


def load_years(years, data_dir="data") -> pd.DataFrame:
    """Load and concatenate raw result CSVs for multiple seasons.

    Reads each data/raw_results_<year>.csv file and stacks them into
    a single DataFrame. These files must have been created by extract_year.py.

    Args:
        years: List of season years to load (e.g. [2018, 2019, ..., 2025])
        data_dir: Directory containing the raw CSV files

    Returns:
        Combined DataFrame with all seasons
    """
    dfs = []
    for y in years:
        path = os.path.join(data_dir, f"raw_results_{y}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}. Run extraction first.")
        dfs.append(pd.read_csv(path))
    df = pd.concat(dfs, ignore_index=True)

    # EventDate is needed for chronological sorting in add_rolling_features()
    if "EventDate" not in df.columns:
        raise ValueError("EventDate missing. Make sure extraction saved EventDate.")
    return df


if __name__ == "__main__":
    # Load all seasons from 2018 to 2025
    # 2018-2023 = training data, 2024 = test data, 2025 = unseen/future data
    years = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
    df_all = load_years(years)
    df_all = add_rolling_features(df_all, window=5)

    # Save the complete feature-engineered dataset
    os.makedirs("data", exist_ok=True)
    out_path = "data/dataset_with_features.csv"
    df_all.to_csv(out_path, index=False)
    print("Saved:", out_path)
