"""
train_models.py
===============
Model training module — the third step in the ML pipeline.

Trains and persists multiple classifiers for F1 top-K finish prediction.

Pipeline position:
  extract_year.py  →  build_dataset.py  →  train_models.py  →  evaluate_models.py

Models trained (3 classifiers × 3 targets = 9 models total):
  - Logistic Regression: simple linear baseline for comparison
  - Random Forest: primary model — handles non-linear feature interactions
  - Gradient Boosting: sequential boosted trees for comparison

Each model is trained for three binary classification targets:
  - Top3 (podium finish), Top5, Top10

Training data: Year <= 2023 (seasons 2018–2023)
Test data (used later in evaluate_models.py): Year == 2024

Key hyperparameters (Random Forest — the primary model):
  - n_estimators=400: number of trees in the forest
  - max_depth=14: limits tree depth to reduce overfitting
  - class_weight="balanced": automatically upweights the minority class
    (only ~15% of drivers finish top-3, so without this the model
    would just predict "not top-3" for everyone)

All models are saved to the models/ directory as .joblib files.
The RF models are also saved under the original rf_top*.joblib names
so the Streamlit app can load them without changes.
"""

import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


# ── Features ─────────────────────────────────────────────────────────

# ── Feature columns ─────────────────────────────────────────────────
# These are the 15 input features the models use for prediction.
# They fall into four groups:
#   1. Categorical: TeamName, EventName (one-hot encoded by the pipeline)
#   2. Grid/qualifying: GridPos, QualiPos (strongest predictors)
#   3. Rolling driver stats: drv_avg_finish_w, drv_top10_rate_w, etc.
#   4. Rolling team stats: team_avg_finish_w, team_top10_rate_w, etc.
FEATURE_COLS = [
    "TeamName", "EventName",               # categorical — one-hot encoded
    "Year",                                 # numeric — captures era effects
    "GridPos", "QualiPos",                  # qualifying/grid — strongest signal
    "career_race_count", "is_rookie",       # experience indicators
    "drv_avg_finish_w", "drv_avg_quali_w",  # driver form (last 5 races)
    "drv_top10_rate_w", "drv_dnf_rate_w",
    "team_avg_finish_w", "team_avg_quali_w",  # team form (last 5 races)
    "team_top10_rate_w", "team_dnf_rate_w",
]

# Split features into categorical (need encoding) and numeric (pass through)
CAT_FEATURES = ["TeamName", "EventName"]
NUM_FEATURES = [c for c in FEATURE_COLS if c not in CAT_FEATURES]

# Binary targets: each model predicts whether a driver finishes in the top K
TARGETS = ["Top3", "Top5", "Top10"]


# ── Pipeline builders ────────────────────────────────────────────────

def _preprocessor():
    """Shared preprocessing: one-hot for categoricals, passthrough for numerics.

    handle_unknown="ignore" ensures new teams/events in the test set
    don't crash the pipeline — they just get all-zero encoded columns.
    """
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
            ("num", "passthrough", NUM_FEATURES),
        ]
    )


def _preprocessor_scaled():
    """Preprocessing with StandardScaler for numeric features.

    Logistic Regression is sensitive to feature scales (it uses gradient
    descent), so numeric features need to be standardised to mean=0, std=1.
    Tree-based models (RF, GB) don't need this — they split on thresholds.
    """
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
            ("num", StandardScaler(), NUM_FEATURES),
        ]
    )


def make_rf_pipeline() -> Pipeline:
    """Random Forest pipeline — the primary model used in the prediction game.

    Hyperparameters:
      - n_estimators=400: ensemble of 400 decision trees (more = better but slower)
      - max_depth=14: prevents overly deep trees that memorise training noise
      - min_samples_split=6, min_samples_leaf=3: regularisation to avoid overfitting
      - class_weight="balanced": handles class imbalance — e.g. only ~15% of
        drivers finish top-3, so the model upweights positive examples
      - random_state=42: ensures reproducible results across runs
      - n_jobs=-1: use all CPU cores for parallel tree training
    """
    return Pipeline([
        ("prep", _preprocessor()),
        ("clf", RandomForestClassifier(
            n_estimators=400,
            max_depth=14,
            min_samples_split=6,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def make_lr_pipeline() -> Pipeline:
    """Logistic Regression pipeline (baseline model)."""
    return Pipeline([
        ("prep", _preprocessor_scaled()),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            solver="lbfgs",
        )),
    ])


def make_gb_pipeline() -> Pipeline:
    """Gradient Boosting pipeline — sequential boosted trees for comparison.

    Unlike RF (parallel trees), GB builds trees sequentially where each
    new tree corrects the mistakes of the previous ones.
    """
    return Pipeline([
        ("prep", _preprocessor()),
        ("clf", GradientBoostingClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )),
    ])


# ── Model registry ──────────────────────────────────────────────────
# Maps short keys to (display name, pipeline builder function) pairs.
# This makes it easy to loop over all models during training/evaluation.

MODEL_BUILDERS = {
    "lr":  ("Logistic Regression", make_lr_pipeline),
    "rf":  ("Random Forest",       make_rf_pipeline),
    "gb":  ("Gradient Boosting",   make_gb_pipeline),
}


# ── Training ─────────────────────────────────────────────────────────

def train_all_models(df: pd.DataFrame, train_end_year: int = 2023) -> dict:
    """
    Train all model types for all targets.

    Returns:
        dict of {model_key: {target: fitted_pipeline}}
        e.g. {"rf": {"Top3": <pipeline>, "Top5": ..., "Top10": ...}, ...}
    """
    train_df = df[df["Year"] <= train_end_year].copy()
    if train_df.empty:
        raise ValueError(f"No training rows with Year <= {train_end_year}")

    missing = [c for c in FEATURE_COLS + TARGETS if c not in train_df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    X_train = train_df[FEATURE_COLS].copy()
    all_models = {}

    for model_key, (model_name, builder_fn) in MODEL_BUILDERS.items():
        all_models[model_key] = {}
        for target in TARGETS:
            y = train_df[target].astype(int).values
            pipe = builder_fn()
            pipe.fit(X_train, y)
            all_models[model_key][target] = pipe
            print(f"  Trained {model_name:25s} | {target} | {len(train_df)} rows")

    return all_models


def save_all_models(all_models: dict, out_dir: str = "models") -> None:
    """Save all trained models as joblib files."""
    os.makedirs(out_dir, exist_ok=True)

    for model_key, targets in all_models.items():
        for target, pipe in targets.items():
            k = target.replace("Top", "")
            filename = f"{model_key}_top{k}.joblib"
            path = os.path.join(out_dir, filename)
            joblib.dump(pipe, path)

    # Also save the RF models under the original rf_top*.joblib filenames
    # so the Streamlit app (app.py) continues to work without changes.
    # The app loads models by these specific paths.
    if "rf" in all_models:
        for target in TARGETS:
            k = target.replace("Top", "")
            original_name = f"rf_top{k}.joblib"
            joblib.dump(all_models["rf"][target], os.path.join(out_dir, original_name))

    print(f"Saved all models to {out_dir}/")


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dataset_path = "data/dataset_with_features.csv"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Could not find {dataset_path}. "
            "Run build_dataset.py first."
        )

    df = pd.read_csv(dataset_path)
    print(f"Loaded {len(df)} rows from {dataset_path}\n")

    all_models = train_all_models(df, train_end_year=2023)
    save_all_models(all_models, out_dir="models")
    print("\nDone.")
