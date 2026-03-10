import os
import fastf1
import pandas as pd

CACHE_DIR = "fastf1_cache"

def init_cache():
    os.makedirs(CACHE_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)


def _format_lap_time(td):
    """Convert a timedelta to a readable lap-time string (e.g. '1:30.456').
    Returns '' for NaT / None values."""
    if pd.isna(td):
        return ""
    total_seconds = td.total_seconds()
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    return f"{minutes}:{seconds:06.3f}"


def get_race_summary(season: int, gp_name: str):
    init_cache()

    session = fastf1.get_session(season, gp_name, "R")
    session.load(weather=True)

    results = session.results[["Abbreviation", "GridPosition", "Position", "Status"]].copy()
    results = results.sort_values("GridPosition").head(20)

    weather = session.weather_data.copy()
    weather_summary = {
        "avg_air_temp": float(weather["AirTemp"].mean()),
        "avg_track_temp": float(weather["TrackTemp"].mean()),
        "max_wind": float(weather["WindSpeed"].max()),
        "rain_any": bool(weather["Rainfall"].any())
    }

    return results, weather_summary


def get_qualifying_results(season: int, gp_name: str) -> pd.DataFrame:
    """Load qualifying session and return lap times per driver."""
    init_cache()

    quali = fastf1.get_session(season, gp_name, "Q")
    quali.load(telemetry=False, weather=False, messages=False)

    qr = quali.results[["Abbreviation", "FirstName", "LastName", "TeamName", "Position", "Q1", "Q2", "Q3"]].copy()
    qr = qr.sort_values("Position").head(20)

    # Build a readable driver name
    qr["Driver"] = qr["FirstName"].str.strip() + " " + qr["LastName"].str.strip()

    # Convert timedeltas to readable strings
    for col in ["Q1", "Q2", "Q3"]:
        qr[col] = qr[col].apply(_format_lap_time)

    qr = qr.rename(columns={"Position": "Pos"})
    qr["Pos"] = qr["Pos"].astype(int, errors="ignore")

    # Build abbreviation → full name mapping for other components
    name_map = dict(zip(qr["Abbreviation"], qr["Driver"]))

    return qr[["Driver", "Abbreviation", "TeamName", "Pos", "Q1", "Q2", "Q3"]].reset_index(drop=True), name_map


def get_circuit_info(season: int, gp_name: str) -> dict:
    """Get circuit metadata from the event schedule."""
    init_cache()

    event = fastf1.get_event(season, gp_name)

    info = {
        "event_name": str(event["EventName"]),
        "country": str(event["Country"]),
        "location": str(event["Location"]),
        "event_date": str(event["EventDate"].strftime("%d %B %Y"))
            if hasattr(event["EventDate"], "strftime") else str(event["EventDate"]),
    }

    # OfficialEventName may contain 'FORMULA 1 ...' with circuit detail
    official = str(event.get("OfficialEventName", ""))
    if official:
        info["official_name"] = official

    return info
