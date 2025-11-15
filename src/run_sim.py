# run_simu_test.py
# First validation experiment:
#   - YasMarina 2017
#   - Valtteri Bottas (BOT)
#   - Fit stint-based linear models from real data
#   - Simulate race with SimpleRaceSim and compare vs real

from __future__ import annotations

import os
import sqlite3
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sim_engine import SimpleRaceSim, StintModel

# Adjust this path if yousim_engine.pyr DB lives somewhere else.
# If this file is inside src/ and the DB is at project root, "../" is correct.
DB_PATH = "../database/F1_timingdata_2014_2019.sqlite"

# ---- user-selectable filters ----
SEASON = 2017
LOCATION_LIKE = "%YasMarina%"   # adjust if needed (use % for LIKE)
DRIVER_CODE = "BOT"             # Bottas as example
PIT_LOSS_SECONDS = 22.0
NOISE_STD = 0.15
# ---------------------------------


def pick_race(conn: sqlite3.Connection, season: int, location_like: str | None) -> int:
    """Choose a race_id for the given season + optional location LIKE filter."""
    seasons = pd.read_sql("SELECT DISTINCT season FROM races ORDER BY season;", conn)
    print("Available seasons:", seasons["season"].tolist())

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
        if not pick.empty:
            race_row = pick.iloc[0]
        else:
            print(f"[warn] No race matching {location_like!r}; using first race of the season.")
            race_row = races.iloc[0]
    else:
        race_row = races.iloc[0]

    race_id = int(race_row.race_id)
    print(f"Chosen race: {race_row.location} ({race_row.season}) — race_id={race_id}")
    return race_id


def load_laps_with_meta(conn: sqlite3.Connection, race_id: int) -> pd.DataFrame:
    """Load laps + driver + race info for a given race_id."""
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


def clean_and_augment(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal cleaning: numeric lap/race times, pit flag, fill compounds."""
    df = df.copy()
    df["lap_time_s"] = pd.to_numeric(df["laptime"], errors="coerce")
    df["race_time_s"] = pd.to_numeric(df["racetime"], errors="coerce")

    df = df[(df["lap_time_s"] > 0) & np.isfinite(df["lap_time_s"])].copy()
    df["is_pit"] = (df["pitstopduration"].fillna(0) > 0)
    df["compound"] = df["compound"].fillna("UNK")
    df["nextcompound"] = df["nextcompound"].fillna("")
    df["tireage"] = pd.to_numeric(df["tireage"], errors="coerce")
    return df


def estimate_pit_losses(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Estimate pit-loss on pit laps vs rolling median pace.
    Not strictly needed for the sim itself, but useful for analysis later.
    """
    df = df.copy()
    df.sort_values(["driver_code", "lap"], inplace=True)
    df["rolling_med"] = (
        df.groupby("driver_code")["lap_time_s"]
          .transform(lambda x: x.rolling(window, min_periods=1, center=True).median())
    )
    df["pit_loss_est_s"] = np.where(
        df["is_pit"],
        df["lap_time_s"] - df["rolling_med"],
        np.nan,
    )
    return df


def build_stints(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Segment stints per driver using (compound, tireage) and pit events.

    New stint starts when:
      - first lap for that driver
      - compound changes
      - tireage decreases (new tyres)
      - previous lap was a pit
    """
    df = df_in.copy()
    df.sort_values(["driver_code", "lap"], inplace=True)
    df["tireage"] = pd.to_numeric(df["tireage"], errors="coerce")
    df["tireage"].fillna(-1, inplace=True)

    def _mark_stints(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("lap").copy()
        g["compound_prev"] = g["compound"].shift(1).fillna(g["compound"])
        g["tireage_prev"] = g["tireage"].shift(1).fillna(g["tireage"])
        g["was_pit_prev"] = g["is_pit"].shift(1).fillna(False)

        new_stint = (
            (g["lap"] == g["lap"].min()) |
            (g["compound"] != g["compound_prev"]) |
            (g["tireage"] < g["tireage_prev"]) |
            (g["was_pit_prev"])
        )
        g["stint_no"] = new_stint.cumsum()
        return g

    df = df.groupby("driver_code", group_keys=False).apply(_mark_stints)

    # per-stint summary
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

    # attach pit info from end-lap of each stint (if pit happened there)
    df_endlaps = (
        df.merge(agg[["driver_code", "stint_no", "end_lap"]],
                 on=["driver_code", "stint_no"], how="inner")
          .query("lap == end_lap")
          [["driver_code", "stint_no", "lap", "is_pit", "pitstopduration",
            "pit_loss_est_s", "nextcompound"]]
          .rename(columns={"lap": "pit_lap"})
    )

    stints = agg.merge(df_endlaps, on=["driver_code", "stint_no"], how="left")

    # order columns nicely
    cols = [
        "season", "location", "driver_code", "stint_no", "compound",
        "start_lap", "end_lap", "total_laps", "avg_lap_s",
        "pit_lap", "pitstopduration", "pit_loss_est_s", "nextcompound",
    ]
    stints = stints.reindex(columns=[c for c in cols if c in stints.columns])
    return stints


def fit_stint_linear_model(stint_laps: pd.DataFrame) -> Tuple[float, float]:
    """
    Fit a simple linear model for a single stint:

        lap_time_s ≈ base_pace + deg_rate * (stint_lap_idx - 1)

    We exclude the pit lap itself and very slow outliers so that the model
    reflects 'normal' race pace rather than in-lap/out-lap spikes.
    """
    d = stint_laps.copy().sort_values("lap")

    # drop pit laps and extremely slow laps (e.g. > median + 3s)
    med = d["lap_time_s"].median()
    d = d[~d["is_pit"]]
    d = d[d["lap_time_s"] < med + 3.0]

    if len(d) == 0:
        # fallback: all laps were weird; just use the original data
        d = stint_laps.copy().sort_values("lap")

    d["stint_idx"] = np.arange(1, len(d) + 1, dtype=float)
    X = d["stint_idx"].values
    y = d["lap_time_s"].values

    if len(d) < 2:
        base = float(np.mean(y))
        return base, 0.0

    # np.polyfit: degree 1 -> [slope, intercept]
    slope, intercept = np.polyfit(X, y, 1)
    base_pace = float(intercept)      # value near stint_idx = 0
    deg_rate = float(slope)
    return base_pace, deg_rate


def main():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"SQLite file not found at {DB_PATH!r}")

    conn = sqlite3.connect(DB_PATH)

    # 1) choose race
    race_id = pick_race(conn, SEASON, LOCATION_LIKE)

    # 2) load + clean laps
    df = load_laps_with_meta(conn, race_id)
    if df.empty:
        raise RuntimeError("No lap rows returned for the chosen race.")

    df = clean_and_augment(df)
    df = estimate_pit_losses(df)

    # 3) filter to chosen driver (e.g., Bottas)
    drv = df[df["driver_code"] == DRIVER_CODE].copy().sort_values("lap")
    if drv.empty:
        raise RuntimeError(f"No laps found for driver {DRIVER_CODE!r}")

    print(f"\nDriver {DRIVER_CODE} — laps {drv['lap'].min()} → {drv['lap'].max()}")
    real_total = drv["lap_time_s"].sum()
    print(f"Real total time: {real_total:.3f} s")

    # 4) build stints and extract this driver's stints
    stints_all = build_stints(df)
    stints_drv = stints_all[stints_all["driver_code"] == DRIVER_CODE].copy()
    print("\nStints for driver:")
    print(stints_drv)

    # 5) fit one linear model per stint
    stint_models: list[StintModel] = []
    for _, row in stints_drv.sort_values("stint_no").iterrows():
        mask = (drv["lap"] >= row["start_lap"]) & (drv["lap"] <= row["end_lap"])
        stint_laps = drv[mask]
        base_pace, deg_rate = fit_stint_linear_model(stint_laps)

        stint_models.append(
            StintModel(
                total_laps=int(row["total_laps"]),
                base_pace=base_pace,
                deg_rate=deg_rate,
            )
        )

    print("\nFitted stint models:")
    for sm in stint_models:
        print(f"  stint: laps={sm.total_laps}, base={sm.base_pace:.3f}s, deg={sm.deg_rate:.4f}s/lap")

    # 6) simulate race
    sim = SimpleRaceSim(
        pit_loss=PIT_LOSS_SECONDS,
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

    # 7) Plot real vs simulated lap traces
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
    plt.show()

    conn.close()


if __name__ == "__main__":
    main()
