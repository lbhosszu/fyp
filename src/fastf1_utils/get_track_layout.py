"""
get_track_layout.py
===================
Track layout image generator.

Uses FastF1 telemetry data (car position from the fastest lap) to
create a bird's-eye view PNG of each circuit. These images are displayed
in the Streamlit app next to the race selection.

Process:
  1. Load the race session and pick the fastest lap
  2. Extract X, Y position data from that lap's telemetry
  3. Interpolate gaps, close the loop, and apply light smoothing
  4. Plot with matplotlib and save as a transparent PNG

Output: src/api/track_layouts/<year>/<event_name>.png
"""

import os
import fastf1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fastf1.Cache.enable_cache("fastf1_cache")


def safe_event_name(event_name: str) -> str:
    """Convert a Grand Prix name to a safe filename.
    e.g. "São Paulo Grand Prix" → "sao_paulo_grand_prix"
    """
    name = event_name.lower()
    name = name.replace(" ", "_")
    name = name.replace(".", "")
    name = name.replace("-", "_")
    return name


def generate_track_layout(year: int, event_name: str, base_dir: str = "src/api/track_layouts"):
    """Generate a track layout PNG from the fastest lap's position data.

    Args:
        year: Season year
        event_name: Grand Prix name
        base_dir: Root directory for track layout images
    """
    year_dir = os.path.join(base_dir, str(year))
    os.makedirs(year_dir, exist_ok=True)

    print(f"Generating track for {event_name} {year}")

    # Load the full race session (needs telemetry for position data)
    session = fastf1.get_session(year, event_name, "R")
    session.load()

    # Use the fastest lap's car position data to trace the circuit shape
    lap = session.laps.pick_fastest()
    pos = lap.get_pos_data()[["X", "Y"]].copy()

    # Interpolate any gaps in the position data (telemetry can have missing points)
    pos["X"] = pos["X"].interpolate(limit_direction="both")
    pos["Y"] = pos["Y"].interpolate(limit_direction="both")
    pos = pos.dropna().reset_index(drop=True)

    # Close the loop: connect the last point back to the first
    pos = pd.concat([pos, pos.iloc[[0]]], ignore_index=True)

    # Light smoothing: take every 2nd point to reduce jaggedness
    pos = pos.iloc[::2].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(pos["X"], pos["Y"], linewidth=3)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    filename = f"{safe_event_name(event_name)}.png"
    filepath = os.path.join(year_dir, filename)

    fig.savefig(filepath, dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)

    print(f"Saved to {filepath}")


if __name__ == "__main__":
    year = 2023
    schedule = fastf1.get_event_schedule(year)

    for event in schedule["EventName"].tolist():
        try:
            generate_track_layout(year, event)
        except Exception as e:
            print(f"Failed for {event}: {e}")
