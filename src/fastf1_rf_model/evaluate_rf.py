import pandas as pd
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix


FEATURE_COLS = [
    "TeamName", "EventName",
    "Year",
    "GridPos", "QualiPos",
    "career_race_count", "is_rookie",
    "drv_avg_finish_w", "drv_avg_quali_w",
    "drv_top10_rate_w", "drv_dnf_rate_w",
    "team_avg_finish_w", "team_avg_quali_w",
    "team_top10_rate_w", "team_dnf_rate_w",
]


def topk_hit_rate(df_race: pd.DataFrame, prob_col: str, k: int, truth_col: str) -> float:
    """
    For one race: sort by predicted prob, take top k.
    Score = (# of true top-k drivers captured) / k
    """
    df_sorted = df_race.sort_values(prob_col, ascending=False)
    pred_topk = df_sorted.head(k)
    return pred_topk[truth_col].mean()  # since truth_col is 1 for true TopK


def evaluate_by_race(df_test: pd.DataFrame, model, target: str, k: int) -> pd.DataFrame:
    X = df_test[FEATURE_COLS].copy()
    probs = model.predict_proba(X)[:, 1]
    df = df_test.copy()
    df[f"P_{target}"] = probs

    rows = []
    for (year, event), g in df.groupby(["Year", "EventName"]):
        score = topk_hit_rate(g, f"P_{target}", k=k, truth_col=target)
        rows.append({"Year": year, "EventName": event, "TopK_HitRate": score})

    out = pd.DataFrame(rows).sort_values(["Year", "TopK_HitRate"], ascending=[True, False])
    return out


if __name__ == "__main__":
    from fastf1_rf_model.train_rf import train_models

    df = pd.read_csv("data/dataset_with_features.csv")

    # Train on <= 2023
    models = train_models(df, train_end_year=2023)

    # Test on 2024
    df_test = df[df["Year"] == 2024].copy()

    print("\nRows in 2024 test:", len(df_test))

    for target, k in [("Top3", 3), ("Top5", 5), ("Top10", 10)]:
        print(f"\n=== {target} (Top{k}) ===")
        # Row-level metrics (not perfect for your game but still useful)
        y_true = df_test[target].astype(int).values
        y_pred = models[target].predict(df_test[FEATURE_COLS])
        print(confusion_matrix(y_true, y_pred))
        print(classification_report(y_true, y_pred, digits=3))

        # Race-level game metric
        by_race = evaluate_by_race(df_test, models[target], target=target, k=k)
        print("\nAverage TopK hit-rate across 2024 races:", by_race["TopK_HitRate"].mean())
        print("Worst 5 races:\n", by_race.tail(5).to_string(index=False))
