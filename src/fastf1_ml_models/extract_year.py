import os
import time
import fastf1
import pandas as pd
import numpy as np

fastf1.Cache.enable_cache("fastf1_cache")


def _safe_int(x):
    try:
        return int(x)
    except Exception:
        return np.nan


def schedule_csv_path(year: int) -> str:
    return os.path.join("schedules", f"f1_schedule_{year}.csv")


def get_schedule(year: int, retries: int = 3) -> pd.DataFrame:
    """
    Try FastF1 schedule. If that fails, fall back to schedules/f1_schedule_<year>.csv
    CSV must contain columns: EventName, EventDate (YYYY-MM-DD)
    """
    last_err = None
    for i in range(retries):
        try:
            return fastf1.get_event_schedule(year)
        except Exception as e:
            last_err = e
            time.sleep(2 * (i + 1))

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

    merged["DidFinish"] = (merged["Status"] == "Finished").astype(int)

    merged["Top3"] = (merged["FinishPos"] <= 3).astype(int)
    merged["Top5"] = (merged["FinishPos"] <= 5).astype(int)
    merged["Top10"] = (merged["FinishPos"] <= 10).astype(int)

    keep = [
        "Year", "EventName", "DriverNumber", "Abbreviation",
        "TeamName", "GridPos", "QualiPos",
        "FinishPos", "DidFinish",
        "Top3", "Top5", "Top10"
    ]
    out = merged[keep].copy()
    out = out.dropna(subset=["GridPos"])
    return out


def extract_year(year: int, out_dir: str = "data") -> str:
    os.makedirs(out_dir, exist_ok=True)
    sched = get_schedule(year)

    # Some schedules include non-race events, but EventName list is still fine
    rows = []
    for ev in sched["EventName"].tolist():
        try:
            df = load_race_quali_rows(year, ev)
            # Attach event date if present
            if "EventDate" in sched.columns:
                ed = sched.loc[sched["EventName"] == ev, "EventDate"].iloc[0]
                df["EventDate"] = pd.to_datetime(ed)
            rows.append(df)
            print(f"OK: {year} - {ev} ({len(df)} drivers)")
        except Exception as e:
            print(f"SKIP: {year} - {ev}: {e}")

    if not rows:
        raise RuntimeError(f"No races extracted for {year}")

    out_path = os.path.join(out_dir, f"raw_results_{year}.csv")
    pd.concat(rows, ignore_index=True).to_csv(out_path, index=False)
    print("Saved:", out_path)
    return out_path


if __name__ == "__main__":
    # Change this to one year at a time while testing
    extract_year(2018)
