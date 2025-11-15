# sim_test.py
# Simple race simulation engine for FYP – data-driven, stint-based model.

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import numpy as np


@dataclass
class StintModel:
    """Parameters describing one stint for one driver."""
    total_laps: int
    base_pace: float      # seconds (intercept)
    deg_rate: float       # seconds per lap (slope)


class SimpleRaceSim:
    """
    Very lightweight race simulator.

    Assumptions
    -----------
    - Stint structure (number of laps per stint) is fixed in advance.
    - Lap time in a stint is modelled as:

          lap_time = base_pace + deg_rate * (stint_lap_index - 1) + noise

    - Between stints we add a constant pit-loss (e.g. 20–22 s).
      This keeps the model simple and easy to tune.
    """

    def __init__(
        self,
        pit_loss: float = 22.0,
        noise_std: float = 0.15,
        random_state: Optional[int] = None,
    ) -> None:
        self.pit_loss = pit_loss
        self.noise_std = noise_std
        self.rng = np.random.default_rng(random_state)

    def run_race(self, stint_models: List[StintModel]) -> Dict[str, Any]:
        """
        Run a race for a single driver.

        Parameters
        ----------
        stint_models:
            List of StintModel in chronological stint order.

        Returns
        -------
        dict with:
            - "lap_times": np.ndarray of per-lap times (seconds)
            - "cum_times": np.ndarray of cumulative race time (seconds)
            - "total_time": float, final race time (seconds)
        """
        all_laps: list[float] = []
        total_time = 0.0

        for i, stint in enumerate(stint_models):
            L = int(stint.total_laps)
            if L <= 0:
                continue

            lap_idx = np.arange(1, L + 1, dtype=float)
            base = stint.base_pace
            deg = stint.deg_rate

            lap_times = base + deg * (lap_idx - 1.0)

            if self.noise_std > 0:
                noise = self.rng.normal(0.0, self.noise_std, size=L)
                lap_times = lap_times + noise

            all_laps.extend(lap_times.tolist())
            total_time += float(lap_times.sum())

            # add pit-loss between stints (not after final stint)
            if i < len(stint_models) - 1:
                total_time += self.pit_loss

        lap_times_arr = np.array(all_laps, dtype=float)
        cum_times = lap_times_arr.cumsum()

        return {
            "lap_times": lap_times_arr,
            "cum_times": cum_times,
            "total_time": float(total_time),
        }
