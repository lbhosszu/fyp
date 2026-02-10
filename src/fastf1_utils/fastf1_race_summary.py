import os
import fastf1
import pandas as pd

CACHE_DIR = "fastf1_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

def race_summary(season: int, gp_name: str):
    session = fastf1.get_session(season, gp_name, "R")
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

if __name__ == "__main__":
    results, w = race_summary(2019, "Abu Dhabi")
    print("Top 10 results:")
    print(results)
    print("\nWeather summary:")
    print(w)
