import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


# -----------------------------
# Features
# -----------------------------

FEATURE_COLS = [
    "TeamName", "EventName",          # categorical
    "Year",                           # numeric
    "GridPos", "QualiPos",
    "career_race_count", "is_rookie",
    "drv_avg_finish_w", "drv_avg_quali_w",
    "drv_top10_rate_w", "drv_dnf_rate_w",
    "team_avg_finish_w", "team_avg_quali_w",
    "team_top10_rate_w", "team_dnf_rate_w",
]

CAT_FEATURES = ["TeamName", "EventName"]
NUM_FEATURES = [c for c in FEATURE_COLS if c not in CAT_FEATURES]


# -----------------------------
# Model
# -----------------------------

def make_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
            ("num", "passthrough", NUM_FEATURES),
        ]
    )

    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=14,
        min_samples_split=6,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline([("prep", pre), ("rf", rf)])


def train_models(df: pd.DataFrame, train_end_year: int = 2023) -> dict:
    train_df = df[df["Year"] <= train_end_year].copy()
    if train_df.empty:
        raise ValueError(f"No training rows found with Year <= {train_end_year}")

    # Basic sanity checks
    missing = [c for c in FEATURE_COLS + ["Top3", "Top5", "Top10"] if c not in train_df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    X = train_df[FEATURE_COLS].copy()

    models = {}
    for target in ["Top3", "Top5", "Top10"]:
        y = train_df[target].astype(int).values
        model = make_pipeline()
        model.fit(X, y)
        models[target] = model
        print(f"Trained {target} on {len(train_df)} rows")

    return models


def save_models(models: dict, out_dir: str = "models") -> None:
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(models["Top3"], os.path.join(out_dir, "rf_top3.joblib"))
    joblib.dump(models["Top5"], os.path.join(out_dir, "rf_top5.joblib"))
    joblib.dump(models["Top10"], os.path.join(out_dir, "rf_top10.joblib"))
    print(f"Saved models to {out_dir}/")


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    dataset_path = "data/dataset_with_features.csv"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Could not find {dataset_path}. "
            "Run build_dataset.py first to generate dataset_with_features.csv"
        )

    df = pd.read_csv(dataset_path)

    models = train_models(df, train_end_year=2023)
    save_models(models, out_dir="models")
    print("Done.")
