"""
extract_year.py
===============
Data extraction module — the first step in the ML pipeline.

This script pulls race results and qualifying data from the FastF1 API
for a given F1 season and merges them into a single CSV file per year.

Pipeline position:
  extract_year.py  →  build_dataset.py  →  train_models.py  →  evaluate_models.py

Key responsibilities:
  - Fetch the race schedule for a season (with CSV fallback if API is down)
  - For each Grand Prix, load race results and qualifying results via FastF1
  - Merge race and qualifying data on DriverNumber to get each driver's
    grid position, qualifying position, and finishing position in one row
  - Create binary target columns (Top3, Top5, Top10) used for classification
  - Save the raw results as data/raw_results_<year>.csv

Output columns include:
  Year, EventName, DriverNumber, Abbreviation, TeamName, GridPos, QualiPos,
  FinishPos, DidFinish, Top3, Top5, Top10
"""

import os
import time
import fastf1
import pandas as pd
import numpy as np

# Enable FastF1's built-in caching to avoid re-downloading session data
# on repeated runs. The cache is stored locally in the fastf1_cache/ folder.
fastf1.Cache.enable_cache("fastf1_cache")


def _safe_int(x):
    """Safely convert a value to int, returning NaN if conversion fails.

    This handles edge cases in FastF1 data where position values can sometimes
    be floats, strings, or missing entirely (e.g. for DNS/DNF drivers).
    """
    try:
        return int(x)
    except Exception:
        return np.nan


def schedule_csv_path(year: int) -> str:
    """Return the path to a local CSV fallback schedule for the given year.

    Used when the FastF1 API is unavailable (e.g. rate-limited or offline).
    The CSV must contain at minimum: EventName and EventDate columns.
    """
    return os.path.join("schedules", f"f1_schedule_{year}.csv")


def get_schedule(year: int, retries: int = 3) -> pd.DataFrame:
    """Fetch the F1 race schedule for a season.

    Tries the FastF1 API first (with exponential backoff retries),
    then falls back to a local CSV file if all API attempts fail.
    This makes the pipeline resilient to network issues.

    Args:
        year: The F1 season year (e.g. 2024)
        retries: Number of API retry attempts before falling back to CSV

    Returns:
        DataFrame with at least EventName and EventDate columns
    """
    last_err = None
    for i in range(retries):
        try:
            return fastf1.get_event_schedule(year)
        except Exception as e:
            last_err = e
            # Exponential backoff: wait 2s, 4s, 6s between retries
            time.sleep(2 * (i + 1))

    # API failed — try the local CSV fallback
    csv_path = schedule_csv_path(year)
    if os.path.exists(csv_path):
        sched = pd.read_csv(csv_path)
        if "EventName" not in sched.columns or "EventDate" not in sched.columns:
            raise ValueError(f"{csv_path} must contain EventName, EventDate")
        sched["EventDate"] = pd.to_datetime(sched["EventDate"])
        return sched

    raise ValueError(
        f"Could not load schedule for {year}. Last error: {last_err}\n"
        f"Fallback file not found: {csv_path}"
    )


def load_race_quali_rows(year: int, event_name: str) -> pd.DataFrame:
    """Load and merge race + qualifying results for one Grand Prix.

    This is the core data extraction function. For a single race weekend it:
      1. Loads the Race session results (finishing positions, grid, status)
      2. Loads the Qualifying session results (Q1/Q2/Q3 positions)
      3. Merges them on DriverNumber so each row has both quali and race data
      4. Creates binary target columns (Top3, Top5, Top10) for classification

    We load with telemetry=False, weather=False, messages=False to speed up
    extraction — we only need the results tables, not lap-by-lap data.

    Args:
        year: Season year
        event_name: Grand Prix name (e.g. "Bahrain Grand Prix")

    Returns:
        DataFrame with one row per driver who started the race
    """
    # Load race session (finishing order, DNF status, etc.)
    race = fastf1.get_session(year, event_name, "R")
    race.load(telemetry=False, weather=False, messages=False)

    # Load qualifying session (grid-determining positions)
    quali = fastf1.get_session(year, event_name, "Q")
    quali.load(telemetry=False, weather=False, messages=False)

    rr = race.results.copy()
    qr = quali.results.copy()

    # Standardise DriverNumber to string for a clean merge key
    rr["DriverNumber"] = rr["DriverNumber"].astype(str)
    qr["DriverNumber"] = qr["DriverNumber"].astype(str)

    # Left-join: keep all race starters, attach their qualifying position
    merged = pd.merge(
        rr,
        qr[["DriverNumber", "Position", "TeamName"]],
        on="DriverNumber",
        how="left",
        suffixes=("_race", "_quali")
    )

    # Rename columns to clear, consistent names used across the pipeline
    merged = merged.rename(columns={
        "Position_race": "FinishPos",
        "Position_quali": "QualiPos",
        "GridPosition": "GridPos",
        "TeamName_race": "TeamName"
    })

    merged["Year"] = int(year)
    merged["EventName"] = event_name

    # Convert positions to integers (handles NaN gracefully)
    merged["FinishPos"] = merged["FinishPos"].apply(_safe_int)
    merged["QualiPos"] = merged["QualiPos"].apply(_safe_int)
    merged["GridPos"] = merged["GridPos"].apply(_safe_int)

    # Binary flag: 1 if the driver completed the race, 0 otherwise (DNF/DNS)
    merged["DidFinish"] = (merged["Status"] == "Finished").astype(int)

    # Binary classification targets — these are what the ML models predict
    # Top3 = podium finish, Top5 = points (roughly), Top10 = solid finish
    merged["Top3"] = (merged["FinishPos"] <= 3).astype(int)
    merged["Top5"] = (merged["FinishPos"] <= 5).astype(int)
    merged["Top10"] = (merged["FinishPos"] <= 10).astype(int)

    # Keep only the columns needed for the ML pipeline
    keep = [
        "Year", "EventName", "DriverNumber", "Abbreviation",
        "TeamName", "GridPos", "QualiPos",
        "FinishPos", "DidFinish",
        "Top3", "Top5", "Top10"
    ]
    out = merged[keep].copy()
    # Drop rows where GridPos is NaN (driver didn't actually start)
    out = out.dropna(subset=["GridPos"])
    return out


def extract_year(year: int, out_dir: str = "data") -> str:
    """Extract race data for an entire season and save to CSV.

    Iterates through every Grand Prix in the schedule, extracts race and
    qualifying data, and concatenates into a single CSV file. Events that
    fail (e.g. cancelled races, sprint-only weekends) are skipped with
    a warning rather than crashing the whole pipeline.

    Args:
        year: The F1 season year to extract
        out_dir: Directory to save the output CSV

    Returns:
        Path to the saved CSV file (e.g. "data/raw_results_2024.csv")
    """
    os.makedirs(out_dir, exist_ok=True)
    sched = get_schedule(year)

    rows = []
    for ev in sched["EventName"].tolist():
        try:
            df = load_race_quali_rows(year, ev)
            # Attach event date if the schedule provides it (needed for
            # chronological sorting in build_dataset.py)
            if "EventDate" in sched.columns:
                ed = sched.loc[sched["EventName"] == ev, "EventDate"].iloc[0]
                df["EventDate"] = pd.to_datetime(ed)
            rows.append(df)
            print(f"OK: {year} - {ev} ({len(df)} drivers)")
        except Exception as e:
            # Skip events that can't be loaded (pre-season tests, cancelled, etc.)
            print(f"SKIP: {year} - {ev}: {e}")

    if not rows:
        raise RuntimeError(f"No races extracted for {year}")

    out_path = os.path.join(out_dir, f"raw_results_{year}.csv")
    pd.concat(rows, ignore_index=True).to_csv(out_path, index=False)
    print("Saved:", out_path)
    return out_path


if __name__ == "__main__":
    # Run extraction for a single year (change as needed)
    # For the full dataset, loop over [2018, 2019, ..., 2025]
    extract_year(2018)
