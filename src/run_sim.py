# run_simulation_test.py
# First validation experiment for the race simulation engine.
#
# Goal:
#   1. Load real lap-time data (e.g., Bottas Yas Marina 2017/2019).
#   2. Segment the race into tyre stints.
#   3. For each stint, fit a very simple linear tyre-degradation model:
#        lap_time = base_pace + deg_rate * lap_index
#   4. Build a sequence of StintModel objects.
#   5. Run SimpleRaceSim with these models to generate synthetic race output.
#   6. Compare real vs simulated lap traces and total time.
#
# This is the first proper validation step for the simulation engine.

from __future__ import annotations

import os
import sqlite3
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# The simulation engine:
# - SimpleRaceSim: executes race with noise + pit losses
# - StintModel: holds (base pace, degradation rate, length of stint)
from sim_engine_test import SimpleRaceSim, StintModel

# Path to TUM F1 timing database
DB_PATH = "../database/F1_timingdata_2014_2019.sqlite"

# ---- experiment configuration ----
SEASON = 2018                 # choose season
LOCATION_LIKE = "%YasMarina%" # filter race by location string
DRIVER_CODE = "BOT"           # which driver to validate against
PIT_LOSS_SECONDS = 22.0       # fallback pit delta if missing
NOISE_STD = 0.15              # small random variation added in simulator
# ----------------------------------


# --------------------------------------------------------------------------
# Utility: save figure to outputs folder
# --------------------------------------------------------------------------
def save_plot(path: str):
    """Create folder if needed and save the current matplotlib figure."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved plot: {path}")


# --------------------------------------------------------------------------
# Pick a race from the DB
# --------------------------------------------------------------------------
def pick_race(conn: sqlite3.Connection, season: int, location_like: str | None) -> int:
    """
    Select a race_id based on season + optional LIKE match on location.
    Lets us easily switch between circuits during testing.
    """
    seasons = pd.read_sql("SELECT DISTINCT season FROM races ORDER BY season;", conn)
    print("Available seasons:", seasons["season"].tolist())

    # Load races for this season
    races = pd.read_sql(
        """
        SELECT id AS race_id, season, date, location, nolaps, tracklength, availablecompounds
        FROM races
        WHERE season = ?
        ORDER BY date
        """,
        conn,
        params=[season],
    )
    if races.empty:
        raise RuntimeError(f"No races found for season {season}")

    # Try a fuzzy match on location
    if location_like:
        pick = pd.read_sql(
            """
            SELECT id AS race_id, season, date, location, nolaps
            FROM races
            WHERE season = ? AND location LIKE ?
            ORDER BY date
            """,
            conn,
            params=[season, location_like],
        )
        # If we found at least one match, use the first
        if not pick.empty:
            race_row = pick.iloc[0]
        else:
            print(f"[warn] No race matching {location_like!r}; using first race of season.")
            race_row = races.iloc[0]
    else:
        race_row = races.iloc[0]

    race_id = int(race_row.race_id)
    print(f"Chosen race: {race_row.location} ({race_row.season}) — race_id={race_id}")
    return race_id


# --------------------------------------------------------------------------
# Load lap-level raw data
# --------------------------------------------------------------------------
def load_laps_with_meta(conn: sqlite3.Connection, race_id: int) -> pd.DataFrame:
    """
    Load complete lap-level dataset for the selected race:
        - lap times
        - driver info
        - compound, tyre age
        - VSC/SC flags
        - pit info
    This is the raw input used to derive stint structure + fit degradation models.
    """
    q = """
    SELECT
      l.race_id                 AS race_id,
      r.season,
      r.date,
      r.location,
      r.nolaps,
      r.tracklength,
      d.id                      AS driver_id,
      d.initials                AS driver_code,
      d.name                    AS driver_name,
      l.lapno                   AS lap,
      l.position,
      l.laptime,
      l.racetime,
      l.gap, l."interval",
      l.compound,
      l.tireage,
      l.pitintime,
      l.pitstopduration,
      l.nextcompound,
      l.startlapprog_vsc, l.endlapprog_vsc, l.age_vsc,
      l.startlapprog_sc,  l.endlapprog_sc,  l.age_sc
    FROM laps l
    JOIN drivers d ON d.id = l.driver_id
    JOIN races   r ON r.id = l.race_id
    WHERE l.race_id = ?
    ORDER BY d.initials, lap
    """
    df = pd.read_sql(q, conn, params=[race_id])
    return df


# --------------------------------------------------------------------------
# Minimal cleaning of raw laps
# --------------------------------------------------------------------------
def clean_and_augment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert text columns → numeric, remove invalid laps, create pit flags,
    fill missing tyre compound, and ensure tyre age numeric.
    """
    df = df.copy()
    df["lap_time_s"] = pd.to_numeric(df["laptime"], errors="coerce")
    df["race_time_s"] = pd.to_numeric(df["racetime"], errors="coerce")

    # Drop laps with invalid numerical data
    df = df[(df["lap_time_s"] > 0) & np.isfinite(df["lap_time_s"])].copy()

    # Pit detection
    df["is_pit"] = (df["pitstopduration"].fillna(0) > 0)

    # Fill missing compound info
    df["compound"] = df["compound"].fillna("UNK")
    df["nextcompound"] = df["nextcompound"].fillna("")

    df["tireage"] = pd.to_numeric(df["tireage"], errors="coerce")
    return df


# --------------------------------------------------------------------------
# Estimate pit-loss values (optional diagnostic)
# --------------------------------------------------------------------------
def estimate_pit_losses(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Estimate pit-loss for each pit lap relative to rolling median pace.
    Used only for analysis, not directly in the sim (we use StintModel.pit_loss).
    """
    df = df.copy()
    df.sort_values(["driver_code", "lap"], inplace=True)

    # Rolling median smooths out pace fluctuations
    df["rolling_med"] = (
        df.groupby("driver_code")["lap_time_s"]
          .transform(lambda x: x.rolling(window, min_periods=1, center=True).median())
    )

    # Pit loss = (pit lap time) - (expected median lap time)
    df["pit_loss_est_s"] = np.where(
        df["is_pit"],
        df["lap_time_s"] - df["rolling_med"],
        np.nan,
    )
    return df


# --------------------------------------------------------------------------
# Stint segmentation logic
# --------------------------------------------------------------------------
def build_stints(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Build a table of stints for all drivers.
    A new stint is triggered when:
      - the driver begins their first lap,
      - the tyre compound changes,
      - tyre age decreases (fresh tyres),
      - or a pit stop occurred on the previous lap.

    Output is a compact summary per stint:
      start_lap, end_lap, compound, total_laps, average pace, pit info, etc.
    """
    df = df_in.copy()
    df.sort_values(["driver_code", "lap"], inplace=True)
    df["tireage"] = pd.to_numeric(df["tireage"], errors="coerce")
    df["tireage"].fillna(-1, inplace=True)

    # Group per driver and assign stint numbers
    def _mark_stints(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("lap").copy()
        g["compound_prev"] = g["compound"].shift(1).fillna(g["compound"])
        g["tireage_prev"] = g["tireage"].shift(1).fillna(g["tireage"])
        g["was_pit_prev"] = g["is_pit"].shift(1).fillna(False)

        # Define new-stint boundaries
        new_stint = (
            (g["lap"] == g["lap"].min()) |
            (g["compound"] != g["compound_prev"]) |
            (g["tireage"] < g["tireage_prev"]) |
            (g["was_pit_prev"])
        )

        # Cumulative sum gives stint numbers
        g["stint_no"] = new_stint.cumsum()
        return g

    df = df.groupby("driver_code", group_keys=False).apply(_mark_stints)

    # Compute summary per stint
    agg = (
        df.groupby(["driver_code", "stint_no"], as_index=False)
          .agg(
              season=("season", "first"),
              location=("location", "first"),
              compound=("compound", lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0]),
              start_lap=("lap", "min"),
              end_lap=("lap", "max"),
              total_laps=("lap", "count"),
              avg_lap_s=("lap_time_s", "mean"),
          )
          .sort_values(["driver_code", "stint_no"])
    )

    # Extract pit info for the end-lap of each stint (if pit happened there)
    df_endlaps = (
        df.merge(
            agg[["driver_code", "stint_no", "end_lap"]],
            on=["driver_code", "stint_no"],
            how="inner"
        )
        .query("lap == end_lap")
        [["driver_code", "stint_no", "lap", "is_pit", "pitstopduration",
          "pit_loss_est_s", "nextcompound"]]
        .rename(columns={"lap": "pit_lap"})
    )

    # Merge pit info into stint summary
    stints = agg.merge(df_endlaps, on=["driver_code", "stint_no"], how="left")

    # Order columns for readability
    cols = [
        "season", "location", "driver_code", "stint_no", "compound",
        "start_lap", "end_lap", "total_laps", "avg_lap_s",
        "pit_lap", "pitstopduration", "pit_loss_est_s", "nextcompound",
    ]
    stints = stints.reindex(columns=[c for c in cols if c in stints.columns])
    return stints


# --------------------------------------------------------------------------
# Fit linear pace model for a stint
# --------------------------------------------------------------------------
def fit_stint_linear_model(stint_laps: pd.DataFrame) -> Tuple[float, float]:
    """
    Fit a simple linear model describing tyre degradation during a stint:

        lap_time_s ≈ base_pace + deg_rate * (lap_index)

    Cleaning:
      - Remove pit laps
      - Remove extremely slow anomaly laps (median + 3s rule)
    """
    d = stint_laps.copy().sort_values("lap")

    # Remove outliers that would distort the fit
    med = d["lap_time_s"].median()
    d = d[~d["is_pit"]]
    d = d[d["lap_time_s"] < med + 3.0]

    # If all laps were filtered out (rare), fall back to raw
    if len(d) == 0:
        d = stint_laps.copy().sort_values("lap")

    # Build regression index (1,2,3,…)
    d["stint_idx"] = np.arange(1, len(d) + 1, dtype=float)
    X = d["stint_idx"].values
    y = d["lap_time_s"].values

    # If only 1 lap exists, slope = 0
    if len(d) < 2:
        base = float(np.mean(y))
        return base, 0.0

    # Fit y = mX + c (deg_rate = m, base = c)
    slope, intercept = np.polyfit(X, y, 1)
    base_pace = float(intercept)
    deg_rate = float(slope)
    return base_pace, deg_rate


# --------------------------------------------------------------------------
# Main experiment logic
# --------------------------------------------------------------------------
def main():
    # Ensure database exists
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"SQLite file not found at {DB_PATH!r}")

    conn = sqlite3.connect(DB_PATH)

    # STEP 1 — choose race to validate
    race_id = pick_race(conn, SEASON, LOCATION_LIKE)

    # STEP 2 — load and clean laps
    df = load_laps_with_meta(conn, race_id)
    if df.empty:
        raise RuntimeError("No lap rows returned for the chosen race.")

    df = clean_and_augment(df)
    df = estimate_pit_losses(df)  # diagnostic info

    # STEP 3 — isolate chosen driver (e.g. Bottas)
    drv = df[df["driver_code"] == DRIVER_CODE].copy().sort_values("lap")
    if drv.empty:
        raise RuntimeError(f"No laps found for driver {DRIVER_CODE!r}")

    print(f"\nDriver {DRIVER_CODE} — laps {drv['lap'].min()} → {drv['lap'].max()}")
    real_total = drv["lap_time_s"].sum()
    print(f"Real total time: {real_total:.3f} s")

    # STEP 4 — build stints & pick this driver's stints
    stints_all = build_stints(df)
    stints_drv = stints_all[stints_all["driver_code"] == DRIVER_CODE].copy()

    print("\nStints for driver:")
    print(stints_drv)

    # STEP 5 — fit linear model per stint → create StintModel objects
    stint_models: list[StintModel] = []
    max_stint_no = int(stints_drv["stint_no"].max())

    for _, row in stints_drv.sort_values("stint_no").iterrows():

        # Extract actual lap rows for this stint
        mask = (drv["lap"] >= row["start_lap"]) & (drv["lap"] <= row["end_lap"])
        stint_laps = drv[mask]

        # Fit degradation model
        base_pace, deg_rate = fit_stint_linear_model(stint_laps)

        # Determine pit loss (no pit after final stint)
        is_last_stint = (int(row["stint_no"]) == max_stint_no)

        if is_last_stint:
            pit_loss_val = None
        else:
            # Prefer real pitstopduration if available
            pl = row.get("pitstopduration", np.nan)
            try:
                pl_val = float(pl)
            except (TypeError, ValueError):
                pl_val = float("nan")

            if np.isnan(pl_val) or pl_val <= 0:
                pl_val = PIT_LOSS_SECONDS

            pit_loss_val = pl_val

        # Build model object
        stint_models.append(
            StintModel(
                total_laps=int(row["total_laps"]),
                base_pace=base_pace,
                deg_rate=deg_rate,
                pit_loss=pit_loss_val,
            )
        )

    print("\nFitted stint models:")
    for sm in stint_models:
        print(
            f"  stint: laps={sm.total_laps}, "
            f"base={sm.base_pace:.3f}s, deg={sm.deg_rate:.4f}s/lap, "
            f"pit_loss={sm.pit_loss}"
        )

    # STEP 6 — run the simulation using these fitted models
    sim = SimpleRaceSim(
        default_pit_loss=PIT_LOSS_SECONDS,
        noise_std=NOISE_STD,
        random_state=42,
    )
    sim_out = sim.run_race(stint_models)
    sim_laps = sim_out["lap_times"]
    sim_total = sim_out["total_time"]

    print(f"\nSimulated total time: {sim_total:.3f} s")
    diff = sim_total - real_total
    pct = (sim_total / real_total - 1.0) * 100.0
    print(f"Difference: {diff:+.3f} s ({pct:+.2f}%)")

    # STEP 7 — plot real vs simulated lap traces (main validation output)
    laps = drv["lap"].values
    real_laps = drv["lap_time_s"].values

    plt.figure(figsize=(10, 4))
    plt.plot(laps, real_laps, label="Real", lw=1.2)
    plt.plot(laps, sim_laps, label="Simulated", lw=1.2, linestyle="--")
    plt.xlabel("Lap")
    plt.ylabel("Lap time (s)")
    title = f"{DRIVER_CODE} — {drv['location'].iat[0]} {int(drv['season'].iat[0])}: real vs simulated"
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    # Save validation figure
    race_tag = f"{drv['location'].iat[0].replace(' ', '_')}_{int(drv['season'].iat[0])}"
    race_dir = os.path.join("../outputs/sim_results/graphs", race_tag)
    img_path = os.path.join(race_dir, f"{race_tag}_{DRIVER_CODE}_real_vs_sim.png")
    save_plot(img_path)

    plt.show()

    conn.close()


if __name__ == "__main__":
    main()
