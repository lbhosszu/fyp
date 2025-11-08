# analyze_f1.py
# F1 SQLite exploration + plotting (fixed merge on race_id)

import os
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DB_PATH = "F1_timingdata_2014_2019.sqlite"

# ---- user-selectable filters ----
SEASON = 2017
LOCATION_LIKE = "%Marina%"   # e.g. "%Bahrain%", "%Monza%", set to None to skip
# ---------------------------------


def pick_race(conn, season: int, location_like: str | None):
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


def load_laps_with_meta(conn, race_id: int) -> pd.DataFrame:
    q = """
    SELECT
      l.race_id,
      r.season, r.date, r.location, r.nolaps, r.tracklength,
      d.id AS driver_id, d.initials AS driver_code, d.name AS driver_name,
      l.lapno AS lap, l.position, l.laptime, l.racetime, l.gap, l."interval",
      l.compound, l.tireage, l.pitintime, l.pitstopduration, l.nextcompound,
      l.startlapprog_vsc, l.endlapprog_vsc, l.age_vsc,
      l.startlapprog_sc, l.endlapprog_sc, l.age_sc
    FROM laps l
    JOIN drivers d ON d.id = l.driver_id
    JOIN races r   ON r.id = l.race_id
    WHERE l.race_id = ?
    ORDER BY d.initials, lap
    """
    df = pd.read_sql(q, conn, params=[race_id])

    starter = pd.read_sql(
        """
        SELECT race_id, driver_id, gridposition, resultposition, team, status
        FROM starterfields
        WHERE race_id = ?
        """,
        conn,
        params=[race_id],
    )

    if "race_id" not in df.columns:
        df["race_id"] = race_id
    df["driver_id"] = df["driver_id"].astype(int, errors="ignore")
    starter["driver_id"] = starter["driver_id"].astype(int, errors="ignore")

    df = df.merge(starter, on=["race_id", "driver_id"], how="left")
    return df


def clean_and_augment(df: pd.DataFrame) -> pd.DataFrame:
    df["lap_time_s"] = pd.to_numeric(df["laptime"], errors="coerce")
    df["race_time_s"] = pd.to_numeric(df["racetime"], errors="coerce")

    df = df[(df["lap_time_s"] > 0) & np.isfinite(df["lap_time_s"])].copy()

    df["is_pit"] = (df["pitstopduration"].fillna(0) > 0)
    df["compound"] = df["compound"].fillna("UNK")
    df["nextcompound"] = df["nextcompound"].fillna("")
    return df


def load_fcy(conn, race_id: int) -> pd.DataFrame:
    return pd.read_sql(
        """
        SELECT startlap AS start_lap, endlap AS end_lap, type
        FROM fcyphases
        WHERE race_id = ?
        ORDER BY startlap
        """,
        conn,
        params=[race_id],
    )


def plot_driver_trace(df: pd.DataFrame, fcy: pd.DataFrame, code: str):
    d = df[df["driver_code"] == code].copy()
    if d.empty:
        print(f"No laps for driver {code}")
        return
    title = f"{code} — {d['location'].iat[0]} {int(d['season'].iat[0])}"

    plt.figure(figsize=(10, 4))
    plt.plot(d["lap"], d["lap_time_s"], lw=1.2, label="lap time")
    plt.scatter(
        d.loc[d["is_pit"], "lap"],
        d.loc[d["is_pit"], "lap_time_s"],
        marker="x", s=50, label="pit"
    )

    if not fcy.empty:
        for i, row in fcy.iterrows():
            plt.axvspan(row["start_lap"], row["end_lap"], alpha=0.15, label=row["type"] if i == 0 else None)

    plt.xlabel("Lap")
    plt.ylabel("Lap time (s)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"SQLite file not found at {DB_PATH!r}")

    conn = sqlite3.connect(DB_PATH)
    race_id = pick_race(conn, SEASON, LOCATION_LIKE)

    df = load_laps_with_meta(conn, race_id)
    if df.empty:
        raise RuntimeError("No lap rows returned for the chosen race.")

    df = clean_and_augment(df)

    # >>> Added: total lap time per driver (in seconds)
    per_driver = (
        df.groupby(["driver_code", "driver_name"])["lap_time_s"]
          .sum()
          .reset_index()
          .sort_values("lap_time_s")
    )
    print("\nTotal lap time per driver (seconds):")
    for _, row in per_driver.iterrows():
        code = row["driver_code"]
        name = row["driver_name"]
        total_s = row["lap_time_s"]
        print(f"  {code:>3} ({name}): {total_s:.3f} s")
    print()

    # quick peek
    print(df[["driver_code", "lap", "lap_time_s", "compound", "tireage", "is_pit"]].head(12))

    fcy = load_fcy(conn, race_id)

    fast4 = (
        df.groupby("driver_code")["lap_time_s"]
          .mean()
          .sort_values()
          .head(4)
          .index
          .tolist()
    )
    print("Fastest by avg lap:", fast4)
    if fast4:
        plot_driver_trace(df, fcy, fast4[0])

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
        plt.show()

    box = df[df["lap"] > 3][["driver_code", "lap_time_s"]].copy()
    if not box.empty:
        box.boxplot(by="driver_code", column="lap_time_s", rot=90)
        plt.suptitle("")
        plt.title("Lap-time distribution by driver")
        plt.xlabel("")
        plt.ylabel("Lap time (s)")
        plt.tight_layout()
        plt.show()

    cols = [
        "season", "date", "location", "nolaps", "tracklength",
        "driver_id,","driver_code", "driver_name",
        "lap", "position", "lap_time_s", "race_time_s",
        "compound", "tireage", "is_pit", "pitstopduration", "nextcompound",
        "gridposition", "resultposition", "race_id"
    ]
    cols = [c for c in cols if c in df.columns]
    df_out = df[cols].sort_values(["driver_code", "lap"]).reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    fname = f"{df['location'].iat[0].replace(' ', '_')}_{int(df['season'].iat[0])}.csv"
    out_path = os.path.join("data", fname)
    df_out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    conn.close()


if __name__ == "__main__":
    main()
