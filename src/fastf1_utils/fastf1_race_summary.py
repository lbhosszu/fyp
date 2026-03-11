"""
fastf1_race_summary.py
======================
Simplified race summary fetcher (earlier version of fastf1_service.py).

This module provides a basic race_summary() function that returns
race results and weather data. It was the original implementation
before fastf1_service.py added qualifying and circuit info support.

Kept in the project as it can be useful for quick standalone testing.
"""

import os
import fastf1
import pandas as pd

CACHE_DIR = "fastf1_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)


def race_summary(season: int, gp_name: str):
    """Fetch race results and weather summary for a Grand Prix.

    Args:
        season: F1 season year (e.g. 2024)
        gp_name: Grand Prix name (e.g. "Abu Dhabi")

    Returns:
        Tuple of (results_df, weather_dict)
    """
    session = fastf1.get_session(season, gp_name, "R")
    session.load(weather=True)

    results = session.results[["Abbreviation", "Position", "Status"]].copy()
    results = results.sort_values("Position").head(20)

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
    print("Top 20 results:")
    print(results)
    print("\nWeather summary:")
    print(w)
