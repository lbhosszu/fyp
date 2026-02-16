import os
import pandas as pd
import numpy as np


def add_rolling_features(df_all: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    df_all["EventDate"] = pd.to_datetime(df_all["EventDate"])
    df = df_all.sort_values(["EventDate", "Year", "EventName"]).copy()

    df["career_race_count"] = df.groupby("DriverNumber").cumcount()
    df["is_rookie"] = (df["career_race_count"] == 0).astype(int)

    gdrv = df.groupby("DriverNumber", group_keys=False)
    df["drv_avg_finish_w"] = gdrv["FinishPos"].apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    df["drv_avg_quali_w"] = gdrv["QualiPos"].apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    df["drv_top10_rate_w"] = gdrv["Top10"].apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    df["drv_dnf_rate_w"] = gdrv["DidFinish"].apply(lambda s: 1.0 - s.shift(1).rolling(window, min_periods=1).mean())

    gteam = df.groupby("TeamName", group_keys=False)
    df["team_avg_finish_w"] = gteam["FinishPos"].apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    df["team_avg_quali_w"] = gteam["QualiPos"].apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    df["team_top10_rate_w"] = gteam["Top10"].apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    df["team_dnf_rate_w"] = gteam["DidFinish"].apply(lambda s: 1.0 - s.shift(1).rolling(window, min_periods=1).mean())

    for col_drv, col_team in [
        ("drv_avg_finish_w", "team_avg_finish_w"),
        ("drv_avg_quali_w", "team_avg_quali_w"),
        ("drv_top10_rate_w", "team_top10_rate_w"),
        ("drv_dnf_rate_w", "team_dnf_rate_w"),
    ]:
        df[col_drv] = df[col_drv].fillna(df[col_team]).fillna(df[col_drv].mean())

    for col in ["team_avg_finish_w", "team_avg_quali_w", "team_top10_rate_w", "team_dnf_rate_w"]:
        df[col] = df[col].fillna(df[col].mean())

    df["QualiPos"] = df["QualiPos"].fillna(df["GridPos"])

    return df


def load_years(years, data_dir="data") -> pd.DataFrame:
    dfs = []
    for y in years:
        path = os.path.join(data_dir, f"raw_results_{y}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}. Run extraction first.")
        dfs.append(pd.read_csv(path))
    df = pd.concat(dfs, ignore_index=True)

    # Ensure EventDate exists (needed for rolling order)
    if "EventDate" not in df.columns:
        raise ValueError("EventDate missing. Make sure extraction saved EventDate.")
    return df


if __name__ == "__main__":
    years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
    df_all = load_years(years)
    df_all = add_rolling_features(df_all, window=5)

    os.makedirs("data", exist_ok=True)
    out_path = "data/dataset_with_features.csv"
    df_all.to_csv(out_path, index=False)
    print("Saved:", out_path)
