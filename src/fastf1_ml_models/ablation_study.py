"""
ablation_study.py
=================
Feature group ablation study for the Random Forest model.

This is a supplementary evaluation script (not part of the main pipeline)
that measures how much each group of features contributes to model accuracy.

Two complementary approaches are used:

  INCREMENTAL ABLATION — "How much does adding each group help?"
    Trains the model with progressively richer feature sets:
      1. Grid only         – GridPos, QualiPos (qualifying baseline)
      2. + Driver history  – Rolling driver performance stats (recent form)
      3. + Team history    – Rolling team performance stats (car performance)
      4. + Context         – TeamName, EventName, Year, experience, rookie flag

  REMOVAL ABLATION — "How much does removing each group hurt?"
    Starts with all features and removes one group at a time.
    This reveals dependencies: if removing driver history hurts more than
    removing team history, driver form is more predictive than car form.

Both approaches report F1-score (row-level) and hit-rate (race-level).
Results are saved as CSVs and bar chart PNGs in the evaluation/ directory.
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


# ── Feature groups ───────────────────────────────────────────────────

GRID_FEATURES = ["GridPos", "QualiPos"]

DRIVER_FEATURES = [
    "drv_avg_finish_w", "drv_avg_quali_w",
    "drv_top10_rate_w", "drv_dnf_rate_w",
]

TEAM_FEATURES = [
    "team_avg_finish_w", "team_avg_quali_w",
    "team_top10_rate_w", "team_dnf_rate_w",
]

CONTEXT_FEATURES = ["TeamName", "EventName", "Year", "career_race_count", "is_rookie"]

ALL_FEATURES = GRID_FEATURES + DRIVER_FEATURES + TEAM_FEATURES + CONTEXT_FEATURES

# Incremental sets: each builds on the previous one, showing marginal gains
INCREMENTAL_SETS = {
    "Grid only":                GRID_FEATURES,
    "+ Driver history":         GRID_FEATURES + DRIVER_FEATURES,
    "+ Team history":           GRID_FEATURES + DRIVER_FEATURES + TEAM_FEATURES,
    "+ Context (all features)": ALL_FEATURES,
}

# Removal sets: remove one group at a time to measure the performance drop
REMOVAL_SETS = {
    "All features":        ALL_FEATURES,
    "− Grid position":     [f for f in ALL_FEATURES if f not in GRID_FEATURES],
    "− Driver history":    [f for f in ALL_FEATURES if f not in DRIVER_FEATURES],
    "− Team history":      [f for f in ALL_FEATURES if f not in TEAM_FEATURES],
    "− Context":           [f for f in ALL_FEATURES if f not in CONTEXT_FEATURES],
}

TARGETS = {"Top3": 3, "Top5": 5, "Top10": 10}

OUT_DIR = "evaluation"


# ── Helpers ──────────────────────────────────────────────────────────

def _identify_cat_num(feature_list):
    """Split feature list into categorical and numeric columns."""
    cat_cols = [f for f in feature_list if f in ("TeamName", "EventName")]
    num_cols = [f for f in feature_list if f not in cat_cols]
    return cat_cols, num_cols


def build_rf_pipeline(feature_list):
    """Build an RF pipeline for the given feature set."""
    cat_cols, num_cols = _identify_cat_num(feature_list)

    transformers = []
    if cat_cols:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        )
    if num_cols:
        transformers.append(("num", "passthrough", num_cols))

    pre = ColumnTransformer(transformers=transformers)

    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=14,
        min_samples_split=6,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline([("prep", pre), ("clf", rf)])


def topk_hit_rate(df_race, prob_col, k, truth_col):
    """Race-level hit-rate: of the top-k predicted, how many are truly top-k."""
    df_sorted = df_race.sort_values(prob_col, ascending=False)
    return df_sorted.head(k)[truth_col].mean()


def evaluate_feature_set(feature_list, train_df, test_df, target, k):
    """Train and evaluate a single feature set / target combination."""
    X_train = train_df[feature_list].copy()
    y_train = train_df[target].astype(int).values
    X_test = test_df[feature_list].copy()
    y_test = test_df[target].astype(int).values

    pipe = build_rf_pipeline(feature_list)
    pipe.fit(X_train, y_train)

    # Row-level F1
    y_pred = pipe.predict(X_test)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Race-level hit-rate
    probs = pipe.predict_proba(X_test)[:, 1]
    tmp = test_df.copy()
    tmp["_prob"] = probs
    hit_rates = []
    for _, g in tmp.groupby(["Year", "EventName"]):
        hr = topk_hit_rate(g, "_prob", k=k, truth_col=target)
        hit_rates.append(hr)

    return f1, np.mean(hit_rates)


# ── Run ablation ─────────────────────────────────────────────────────

def run_ablation(feature_sets: dict, train_df, test_df, label: str) -> pd.DataFrame:
    """Run ablation across all feature sets and targets."""
    rows = []
    for set_name, feat_list in feature_sets.items():
        for target, k in TARGETS.items():
            f1, hit_rate = evaluate_feature_set(feat_list, train_df, test_df, target, k)
            rows.append({
                "Feature Set": set_name,
                "Target": f"Top-{k}",
                "F1": f1,
                "Hit-Rate": hit_rate,
                "Num Features": len(feat_list),
            })
            print(f"  {set_name:30s} | Top-{k} | F1={f1:.3f} | Hit-Rate={hit_rate:.3f}")
    return pd.DataFrame(rows)


# ── Plotting ─────────────────────────────────────────────────────────

def plot_incremental(df: pd.DataFrame, out_dir: str = OUT_DIR):
    """
    Plot incremental ablation: shows how performance improves
    as more feature groups are added.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    feature_sets = df["Feature Set"].unique()
    targets = df["Target"].unique()
    x = np.arange(len(feature_sets))
    width = 0.25
    colors = ["#ef4444", "#f59e0b", "#22c55e"]

    for ax, metric, title in zip(axes, ["F1", "Hit-Rate"],
                                  ["F1 Score", "Race-Level Hit-Rate"]):
        for i, target in enumerate(targets):
            subset = df[df["Target"] == target]
            vals = [subset[subset["Feature Set"] == s][metric].values[0] for s in feature_sets]
            bars = ax.bar(x + i * width, vals, width, label=target,
                          color=colors[i], edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_ylabel(title, fontsize=12)
        ax.set_xticks(x + width)
        ax.set_xticklabels(feature_sets, fontsize=10, rotation=15, ha="right")
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Incremental Feature Ablation — Random Forest",
                 fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(out_dir, "ablation_incremental.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_removal(df: pd.DataFrame, out_dir: str = OUT_DIR):
    """
    Plot removal ablation: shows performance drop when each
    feature group is removed.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    feature_sets = df["Feature Set"].unique()
    targets = df["Target"].unique()
    x = np.arange(len(feature_sets))
    width = 0.25
    colors = ["#ef4444", "#f59e0b", "#22c55e"]

    for ax, metric, title in zip(axes, ["F1", "Hit-Rate"],
                                  ["F1 Score", "Race-Level Hit-Rate"]):
        # Get baseline (all features) values
        baseline_vals = {}
        for target in targets:
            baseline_vals[target] = df[(df["Feature Set"] == "All features") &
                                       (df["Target"] == target)][metric].values[0]

        for i, target in enumerate(targets):
            subset = df[df["Target"] == target]
            vals = [subset[subset["Feature Set"] == s][metric].values[0] for s in feature_sets]
            bars = ax.bar(x + i * width, vals, width, label=target,
                          color=colors[i], edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        # Draw baseline reference line
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)

        ax.set_ylabel(title, fontsize=12)
        ax.set_xticks(x + width)
        ax.set_xticklabels(feature_sets, fontsize=10, rotation=15, ha="right")
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Feature Removal Ablation — Random Forest",
                 fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(out_dir, "ablation_removal.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    dataset_path = "data/dataset_with_features.csv"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Missing {dataset_path}")

    df = pd.read_csv(dataset_path)
    train_df = df[df["Year"] <= 2023].copy()
    test_df = df[df["Year"] == 2024].copy()
    print(f"Train: {len(train_df)} rows (2018-2023)")
    print(f"Test:  {len(test_df)} rows (2024)\n")

    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Incremental ablation ──
    print("=" * 60)
    print("INCREMENTAL ABLATION")
    print("Adding feature groups one at a time")
    print("=" * 60)
    inc_df = run_ablation(INCREMENTAL_SETS, train_df, test_df, "incremental")
    inc_df.to_csv(os.path.join(OUT_DIR, "ablation_incremental.csv"), index=False)

    # ── Removal ablation ──
    print(f"\n{'=' * 60}")
    print("REMOVAL ABLATION")
    print("Removing one feature group at a time")
    print("=" * 60)
    rem_df = run_ablation(REMOVAL_SETS, train_df, test_df, "removal")
    rem_df.to_csv(os.path.join(OUT_DIR, "ablation_removal.csv"), index=False)

    # ── Plots ──
    print(f"\n{'=' * 60}")
    print("Generating plots...")
    print("=" * 60)
    plot_incremental(inc_df)
    plot_removal(rem_df)

    print(f"\nAll ablation results saved to {OUT_DIR}/")
    print("Done.")
