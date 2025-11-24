# analyze_f1.py
# Exploratory analysis of the TUM F1 Timing Database + plotting utilities.
# This script loads a race, cleans the lap data, merges metadata (drivers, grid, pit stops),
# identifies interesting drivers, and generates diagnostic visualisations
# that will later feed into the validation of the race-simulation engine.

import os
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Path to the SQLite database (relative to /src/ in the FYP structure)
DB_PATH = "../database/F1_timingdata_2014_2019.sqlite"

# ---- user-selectable filters ----
# These allow quickly switching season/race for exploratory work.
SEASON = 2019
LOCATION_LIKE = "%Marina%"   #e.g. "%Bahrain%", "%Monza%", set to None for first race
# ---------------------------------

# ---- utility functions ----
def save_plot(path: str):
    """
    Save the current matplotlib figure to disk.
    - Automatically creates directories if missing.
    - Used for exporting artefacts into the /outputs folder.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved plot: {path}")
# ---------------------------------


def pick_race(conn, season: int, location_like: str | None):
    """
    Select a race for the given season.
    1. List available seasons and races.
    2. If LOCATION_LIKE is provided, try a LIKE-based match.
    3. Otherwise fall back to the first race of the season.
    This makes exploratory iteration much easier.
    """
    # List seasons available in DB
    seasons = pd.read_sql("SELECT DISTINCT season FROM races ORDER BY season;", conn)
    print("Available seasons:", seasons["season"].tolist())

    # Retrieve all races for the selected season
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

    # Try to select by approximate location
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
            # If no match, default gracefully
            print(f"[warn] No race matching {location_like!r}; using first race of the season.")
            race_row = races.iloc[0]
    else:
        race_row = races.iloc[0]

    race_id = int(race_row.race_id)
    print(f"Chosen race: {race_row.location} ({race_row.season}) — race_id={race_id}")
    return race_id


def load_laps_with_meta(conn, race_id: int) -> pd.DataFrame:
    """
    Load all lap-level data for one race.
    Also merge:
        - driver info (name, initials)
        - race metadata
        - starterfield metadata (grid, finishing position, team, status)
    The output is a complete lap-by-lap dataset used for validation and visualisation.
    """
    # Main lap query with joins to drivers + races
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

    # Starter/grid info for the race
    starter = pd.read_sql(
        """
        SELECT race_id, driver_id, gridposition, resultposition, team, status
        FROM starterfields
        WHERE race_id = ?
        """,
        conn,
        params=[race_id],
    )

    # Ensure consistent types for merging
    df["driver_id"] = df["driver_id"].astype(int, errors="ignore")
    starter["driver_id"] = starter["driver_id"].astype(int, errors="ignore")

    # Merge lap-level data with starterfield metadata
    df = df.merge(starter, on=["race_id", "driver_id"], how="left")
    return df


def clean_and_augment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw lap data:
    - Convert lap/race times to numeric (SQLite sometimes stores them as text)
    - Remove invalid laps (e.g., zero or negative times)
    - Detect pit stop laps
    - Fill missing compound information
    """
    df["lap_time_s"] = pd.to_numeric(df["laptime"], errors="coerce")
    df["race_time_s"] = pd.to_numeric(df["racetime"], errors="coerce")

    # Remove non-lap rows (safety)
    df = df[(df["lap_time_s"] > 0) & np.isfinite(df["lap_time_s"])].copy()

    # Pit stop flag
    df["is_pit"] = (df["pitstopduration"].fillna(0) > 0)

    # Compounds occasionally missing in DB
    df["compound"] = df["compound"].fillna("UNK")
    df["nextcompound"] = df["nextcompound"].fillna("")

    return df


def load_fcy(conn, race_id: int) -> pd.DataFrame:
    """
    Load FCY/VSC/SC phases for shading the lap-time plot.
    This helps identify anomalies or changes in lap pace.
    """
    fcy = pd.read_sql(
        """
        SELECT startlap AS start_lap, endlap AS end_lap, type
        FROM fcyphases
        WHERE race_id = ?
        ORDER BY startlap
        """,
        conn,
        params=[race_id],
    )
    return fcy


def plot_driver_trace(df: pd.DataFrame, fcy: pd.DataFrame, code: str):
    """
    Plot the lap-time trace for one driver.
    Includes:
      - pit stop markers
      - FCY/SC shading
    Useful for validating tyre-degradation behaviour and extracting patterns
    for the simulation models.
    """
    d = df[df["driver_code"] == code].copy()
    if d.empty:
        print(f"No laps for driver {code}")
        return

    title = f"{code} — {d['location'].iat[0]} {int(d['season'].iat[0])}"

    plt.figure(figsize=(10, 4))
    plt.plot(d["lap"], d["lap_time_s"], lw=1.2, label="lap time")

    # Mark pit laps with an X
    plt.scatter(
        d.loc[d["is_pit"], "lap"],
        d.loc[d["is_pit"], "lap_time_s"],
        marker="x",
        s=50,
        label="pit"
    )

    # Shade yellow flag phases for better context
    if not fcy.empty:
        for i, row in fcy.iterrows():
            plt.axvspan(row["start_lap"], row["end_lap"], alpha=0.15,
                        label=row["type"] if i == 0 else None)

    plt.xlabel("Lap")
    plt.ylabel("Lap time (s)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    # Save into per-race folder
    race_tag = f"{d['location'].iat[0].replace(' ', '_')}_{int(d['season'].iat[0])}"
    race_dir = os.path.join("../outputs/analyze_f1/graphs", race_tag)
    img_path = os.path.join(race_dir, f"{race_tag}_{code}_trace.png")
    save_plot(img_path)

    plt.show()


def main():
    """
    Main pipeline for the race analysis workflow:
      1. Connect to DB
      2. Pick a race
      3. Load & merge lap-level data
      4. Clean and preprocess
      5. Identify top-performing drivers
      6. Produce diagnostic plots:
         - Single-driver trace
         - Fastest-4 comparison
         - Boxplot of lap distributions
      7. Export cleaned structured data for later modelling
    """
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"SQLite file not found at {DB_PATH!r}")

    conn = sqlite3.connect(DB_PATH)

    # --- 1) select race ---
    race_id = pick_race(conn, SEASON, LOCATION_LIKE)

    # --- 2) load raw laps & metadata ---
    df = load_laps_with_meta(conn, race_id)
    if df.empty:
        raise RuntimeError("No lap rows returned for the chosen race.")

    # --- 3) preprocess / tidy dataset ---
    df = clean_and_augment(df)

    # Debug preview of first few rows
    print(df[["driver_code", "lap", "lap_time_s", "compound", "tireage", "is_pit"]].head(12))

    # --- 4) load FCY data for plotting ---
    fcy = load_fcy(conn, race_id)

    # --- 5) find fastest 4 drivers by average lap ---
    fast4 = (
        df.groupby("driver_code")["lap_time_s"]
          .mean()
          .sort_values()
          .head(4)
          .index
          .tolist()
    )
    print("Fastest by avg lap:", fast4)

    # --- 6A) plot single fastest driver ---
    if fast4:
        plot_driver_trace(df, fcy, fast4[0])

    # --- 6B) compare top-4 drivers on the same graph ---
    if len(fast4) >= 2:
        plt.figure(figsize=(10, 4))
        for code in fast4:
            d = df[df["driver_code"] == code]
            plt.plot(d["lap"], d["lap_time_s"], label=code, lw=1.0)

        plt.xlabel("Lap")
        plt.ylabel("Lap time (s)")
        plt.title(f"Lap traces — {df['location'].iat[0]} {int(df['season'].iat[0])}")
        plt.legend(ncol=len(fast4))
        plt.tight_layout()

        # Save fast-4 comparison figure
        race_tag = f"{df['location'].iat[0].replace(' ', '_')}_{int(df['season'].iat[0])}"
        race_dir = os.path.join("../outputs/analyze_f1/graphs", race_tag)
        img_path = os.path.join(race_dir, f"{race_tag}_fast4.png")
        save_plot(img_path)

        plt.show()

    # --- 6C) boxplot of lap distributions per driver ---
    # Excluding first few laps reduces misleading outliers (tyre warm-up etc.)
    box = df[df["lap"] > 3][["driver_code", "lap_time_s"]].copy()
    if not box.empty:
        box.boxplot(by="driver_code", column="lap_time_s", rot=90)
        plt.suptitle("")
        plt.title("Lap-time distribution by driver")
        plt.ylabel("Lap time (s)")
        plt.tight_layout()

        race_tag = f"{df['location'].iat[0].replace(' ', '_')}_{int(df['season'].iat[0])}"
        race_dir = os.path.join("../outputs/analyze_f1/graphs", race_tag)
        img_path = os.path.join(race_dir, f"{race_tag}_boxplot.png")
        save_plot(img_path)

        plt.show()

    # --- 7) export clean CSV for modelling & debugging ---
    cols = [
        "season", "date", "location", "nolaps", "tracklength",
        "driver_id", "driver_code", "driver_name",
        "lap", "position", "lap_time_s", "race_time_s",
        "compound", "tireage", "is_pit", "pitstopduration", "nextcompound",
        "gridposition", "resultposition", "race_id"
    ]
    cols = [c for c in cols if c in df.columns]  # keep only what exists
    df_out = df[cols].sort_values(["driver_code", "lap"]).reset_index(drop=True)

    os.makedirs("../outputs/analyze_f1/race_results", exist_ok=True)
    fname = f"{df['location'].iat[0].replace(' ', '_')}_{int(df['season'].iat[0])}.csv"
    out_path = os.path.join("../outputs/analyze_f1/race_results", fname)
    df_out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    conn.close()


if __name__ == "__main__":
    main()
