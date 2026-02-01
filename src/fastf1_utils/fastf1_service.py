import os
import fastf1_utils

CACHE_DIR = "fastf1_cache"

def init_cache():
    os.makedirs(CACHE_DIR, exist_ok=True)
    fastf1_utils.Cache.enable_cache(CACHE_DIR)

def get_race_summary(season: int, gp_name: str):
    init_cache()

    session = fastf1_utils.get_session(season, gp_name, "R")
    session.load(weather=True)

    results = session.results[["Abbreviation", "Position", "Status"]].copy()
    results = results.sort_values("Position").head(10)

    weather = session.weather_data.copy()
    weather_summary = {
        "avg_air_temp": float(weather["AirTemp"].mean()),
        "avg_track_temp": float(weather["TrackTemp"].mean()),
        "max_wind": float(weather["WindSpeed"].max()),
        "rain_any": bool(weather["Rainfall"].any())
    }

    return results, weather_summary
