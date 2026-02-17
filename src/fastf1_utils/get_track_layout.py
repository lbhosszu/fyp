import os
import fastf1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fastf1.Cache.enable_cache("fastf1_cache")


def safe_event_name(event_name: str) -> str:
    name = event_name.lower()
    name = name.replace(" ", "_")
    name = name.replace(".", "")
    name = name.replace("-", "_")
    return name


def generate_track_layout(year: int, event_name: str, base_dir: str = "src/api/track_layouts"):
    year_dir = os.path.join(base_dir, str(year))
    os.makedirs(year_dir, exist_ok=True)

    print(f"Generating track for {event_name} {year}")

    session = fastf1.get_session(year, event_name, "R")
    session.load()

    lap = session.laps.pick_fastest()
    pos = lap.get_pos_data()[["X", "Y"]].copy()

    # Fix gaps
    pos["X"] = pos["X"].interpolate(limit_direction="both")
    pos["Y"] = pos["Y"].interpolate(limit_direction="both")
    pos = pos.dropna().reset_index(drop=True)

    # Close loop
    pos = pd.concat([pos, pos.iloc[[0]]], ignore_index=True)

    # Light smoothing
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
