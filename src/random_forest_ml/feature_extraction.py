import os
import sqlite3
from pathlib import Path

import pandas as pd
import numpy as np
BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "database" / "F1_timingdata_2014_2019.sqlite"
OUT_PATH = BASE_DIR / "ml_outputs" / "ml_features.csv"

# Load lap data from the database 
def load_laps(conn):
    q = """
    SELECT
        l.race_id, l.lapno AS lap, l.laptime, l.racetime, l.position,
        l.compound, l.tireage, l.pitstopduration,
        d.initials AS driver_code, d.name AS driver_name, d.id AS driver_id,
        r.season, r.location, r.nolaps, r.tracklength
    FROM laps l
    JOIN drivers d ON d.id = l.driver_id
    JOIN races r   ON r.id = l.race_id
    ORDER BY season, race_id, driver_code, lap
    """
    df = pd.read_sql(q, conn)
    df["lap_time_s"] = pd.to_numeric(df["laptime"], errors="coerce")
    df["is_pit"] = (df["pitstopduration"].fillna(0) > 0)
    return df


# Load grid + result positions
def load_grid(conn):
    q = """
    SELECT
        race_id, driver_id, gridposition, resultposition, team
    FROM starterfields
    """
    return pd.read_sql(q, conn)


# Build driver-level features
def build_features(df, grid):
    # Merge grid + results
    df = df.merge(grid, on=["race_id", "driver_id"], how="left")

    # Group by race + driver
    features = []

    for (race_id, driver_id), g in df.groupby(["race_id", "driver_id"]):
        g = g.sort_values("lap")

        # Basic metadata
        season = int(g["season"].iloc[0])
        location = g["location"].iloc[0]
        driver_code = g["driver_code"].iloc[0]
        team = g["team"].iloc[0] if "team" in g.columns else None
        grid_pos = g["gridposition"].iloc[0]
        result_pos = g["resultposition"].iloc[0]

        # Pace features
        avg_lap = g["lap_time_s"].mean()
        median_lap = g["lap_time_s"].median()
        std_lap = g["lap_time_s"].std()
        best_lap = g["lap_time_s"].min()

        # Pit features
        pit_count = g["is_pit"].sum()
        avg_pit_loss = g.loc[g["is_pit"], "pitstopduration"].mean() if pit_count > 0 else 0

        # Tyre features
        compounds_used = g["compound"].dropna().unique()
        tyre_mix = len(compounds_used)

        # Stint features
        g["tireage_shift"] = g["tireage"].shift(1)
        new_stint = (g["tireage"] < g["tireage_shift"]).fillna(True)
        stint_no = new_stint.cumsum()
        stint_lengths = g.groupby(stint_no)["lap"].count().tolist()
        avg_stint_length = np.mean(stint_lengths) if stint_lengths else 0

        # Track metadata
        nolaps = g["nolaps"].iloc[0]
        tracklength = g["tracklength"].iloc[0]

        features.append({
            "race_id": race_id,
            "season": season,
            "location": location,
            "driver_id": driver_id,
            "driver_code": driver_code,
            "team": team,

            # Model target
            "resultposition": result_pos,

            # Starting position
            "gridposition": grid_pos,

            # Pace features
            "avg_lap": avg_lap,
            "median_lap": median_lap,
            "std_lap": std_lap,
            "best_lap": best_lap,

            # Pit features
            "pit_count": pit_count,
            "avg_pit_loss": avg_pit_loss,

            # Tyre features
            "tyre_mix": tyre_mix,

            # Stint features
            "avg_stint_length": avg_stint_length,

            # Track metadata
            "nolaps": nolaps,
            "tracklength": tracklength,
        })

    return pd.DataFrame(features)


def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"SQLite file not found at {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)

    print("Loading data...")
    df = load_laps(conn)
    grid = load_grid(conn)

    print("Building features...")
    features_df = build_features(df, grid)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(OUT_PATH, index=False)

    print(f"Saved ML dataset to: {OUT_PATH}")
    print(features_df.head())


if __name__ == "__main__":
    main()
