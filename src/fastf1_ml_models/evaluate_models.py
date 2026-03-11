"""
evaluate_models.py
==================
Model evaluation module — the final step in the ML pipeline.

Pipeline position:
  extract_year.py  →  build_dataset.py  →  train_models.py  →  evaluate_models.py

Tests all trained models against the 2024 season (held-out test set)
and produces comprehensive evaluation outputs:

  1. Row-level metrics: accuracy, precision, recall, F1 per model × target
     (standard sklearn classification metrics — one row = one driver entry)

  2. Race-level hit-rate: the most GAME-RELEVANT metric. For each race,
     sorts drivers by predicted probability, takes the top K, and checks
     how many are actually in the real top K. This directly maps to how
     well the model would perform in the prediction game.

  3. Per-race breakdown: hit-rate for every 2024 GP — helps identify which
     races are easy/hard to predict (e.g. wet races tend to be harder)

  4. Feature importance: which features matter most to the RF and GB models
     (GridPos/QualiPos typically dominate, validating that qualifying is
     the strongest predictor of race results)

  5. Model comparison charts: bar charts for visual comparison

  6. Per-race heatmap: colour-coded grid showing RF performance at each GP

All outputs (CSVs and PNGs) are saved to the evaluation/ directory.
"""

import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# ── Configuration ────────────────────────────────────────────────────

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

TARGETS = {
    "Top3": 3,
    "Top5": 5,
    "Top10": 10,
}

MODEL_NAMES = {
    "lr": "Logistic Regression",
    "rf": "Random Forest",
    "gb": "Gradient Boosting",
}

OUT_DIR = "evaluation"


# ── Helpers ──────────────────────────────────────────────────────────

def load_models(model_dir: str = "models") -> dict:
    """Load all trained models from the models/ directory."""
    all_models = {}
    for model_key in MODEL_NAMES:
        all_models[model_key] = {}
        for target, k in TARGETS.items():
            path = os.path.join(model_dir, f"{model_key}_top{k}.joblib")
            if os.path.exists(path):
                all_models[model_key][target] = joblib.load(path)
            else:
                print(f"  Warning: {path} not found, skipping.")
    return all_models


def topk_hit_rate(df_race: pd.DataFrame, prob_col: str, k: int, truth_col: str) -> float:
    """Calculate the top-K hit-rate for a single race.

    This is the core game-relevant metric. It simulates what the prediction
    game does: rank all drivers by the model's predicted probability of
    finishing in the top K, pick the top K predictions, and check how many
    of those K drivers actually finished in the top K.

    Example: If k=3 and the model predicts VER, NOR, LEC as top-3, but
    the actual top-3 is VER, LEC, PIA → hit-rate = 2/3 = 0.667

    Args:
        df_race: DataFrame with one row per driver for a single race
        prob_col: Column containing the model's predicted probabilities
        k: Number of top positions to consider (3, 5, or 10)
        truth_col: Binary column (1 if truly in top K, 0 otherwise)

    Returns:
        Float between 0 and 1 — proportion of correct top-K predictions
    """
    df_sorted = df_race.sort_values(prob_col, ascending=False)
    pred_topk = df_sorted.head(k)
    # .mean() of a binary column = proportion of 1s = hit-rate
    return pred_topk[truth_col].mean()


# ── 1. Row-level classification metrics ──────────────────────────────

def row_level_comparison(all_models: dict, df_test: pd.DataFrame) -> pd.DataFrame:
    """
    Compute accuracy, precision, recall, F1 for each model × target.
    Returns a DataFrame for easy display and export.
    """
    X_test = df_test[FEATURE_COLS].copy()
    rows = []

    for model_key, model_name in MODEL_NAMES.items():
        if model_key not in all_models:
            continue
        for target, k in TARGETS.items():
            if target not in all_models[model_key]:
                continue

            model = all_models[model_key][target]
            y_true = df_test[target].astype(int).values
            y_pred = model.predict(X_test)

            rows.append({
                "Model": model_name,
                "Target": f"Top-{k}",
                "Accuracy": accuracy_score(y_true, y_pred),
                "Precision": precision_score(y_true, y_pred, zero_division=0),
                "Recall": recall_score(y_true, y_pred, zero_division=0),
                "F1": f1_score(y_true, y_pred, zero_division=0),
            })

    return pd.DataFrame(rows)


# ── 2. Race-level Top-K hit-rate ─────────────────────────────────────

def race_level_comparison(all_models: dict, df_test: pd.DataFrame) -> pd.DataFrame:
    """Compare models using the race-level top-K hit-rate metric.

    For each model × target combination, computes the hit-rate for
    every race in the 2024 test set, then reports the average, min, max,
    and standard deviation. This is the most important evaluation metric
    because it directly reflects how the model performs in the game context.

    Why race-level and not row-level? Row-level F1 treats every driver
    entry equally, but we care about ranking within each race. A model
    could have high F1 but still rank drivers poorly within a single GP.
    """
    X_test = df_test[FEATURE_COLS].copy()
    rows = []

    for model_key, model_name in MODEL_NAMES.items():
        if model_key not in all_models:
            continue
        for target, k in TARGETS.items():
            if target not in all_models[model_key]:
                continue

            model = all_models[model_key][target]
            probs = model.predict_proba(X_test)[:, 1]
            df_tmp = df_test.copy()
            df_tmp["_prob"] = probs

            hit_rates = []
            for (year, event), g in df_tmp.groupby(["Year", "EventName"]):
                hr = topk_hit_rate(g, "_prob", k=k, truth_col=target)
                hit_rates.append(hr)

            rows.append({
                "Model": model_name,
                "Target": f"Top-{k}",
                "Avg Hit-Rate": np.mean(hit_rates),
                "Min Hit-Rate": np.min(hit_rates),
                "Max Hit-Rate": np.max(hit_rates),
                "Std": np.std(hit_rates),
            })

    return pd.DataFrame(rows)


# ── 3. Per-race breakdown for best model ─────────────────────────────

def per_race_breakdown(all_models: dict, df_test: pd.DataFrame, model_key: str = "rf") -> pd.DataFrame:
    """
    Show the hit-rate for every race in the test set across all targets
    for a given model. Helps identify which GPs are hard/easy to predict.
    """
    if model_key not in all_models:
        return pd.DataFrame()

    X_test = df_test[FEATURE_COLS].copy()
    race_rows = []

    for (year, event), g in df_test.groupby(["Year", "EventName"]):
        row = {"Year": year, "EventName": event}
        for target, k in TARGETS.items():
            if target not in all_models[model_key]:
                continue
            model = all_models[model_key][target]
            probs = model.predict_proba(g[FEATURE_COLS])[:, 1]
            g_copy = g.copy()
            g_copy["_prob"] = probs
            hr = topk_hit_rate(g_copy, "_prob", k=k, truth_col=target)
            row[f"Top-{k} Hit-Rate"] = hr
        race_rows.append(row)

    return pd.DataFrame(race_rows).sort_values("EventName")


# ── 4. Feature importance ────────────────────────────────────────────

def get_feature_names(model_pipeline) -> list:
    """Extract feature names after one-hot encoding.

    After one-hot encoding, TeamName and EventName expand into many
    binary columns (e.g. TeamName_Red Bull Racing, EventName_Bahrain).
    This function recovers those expanded names for feature importance plots.
    """
    prep = model_pipeline.named_steps["prep"]
    cat_encoder = prep.named_transformers_["cat"]
    cat_names = list(cat_encoder.get_feature_names_out(["TeamName", "EventName"]))

    num_names = [c for c in FEATURE_COLS if c not in ["TeamName", "EventName"]]
    return cat_names + num_names


def plot_feature_importance_rf(all_models: dict, out_dir: str = OUT_DIR):
    """
    Plot feature importance for the Random Forest models.
    Shows top 20 features for each target.
    """
    if "rf" not in all_models:
        print("No Random Forest models found, skipping feature importance.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    fig.suptitle("Random Forest — Feature Importance (Top 20)", fontsize=16, fontweight="bold")

    for ax, (target, k) in zip(axes, TARGETS.items()):
        if target not in all_models["rf"]:
            continue

        model = all_models["rf"][target]
        clf = model.named_steps["clf"]
        feat_names = get_feature_names(model)
        importances = clf.feature_importances_

        # Sort and take top 20
        indices = np.argsort(importances)[-20:]
        top_names = [feat_names[i] for i in indices]
        top_vals = importances[indices]

        ax.barh(range(len(top_names)), top_vals, color="#2563eb", edgecolor="white")
        ax.set_yticks(range(len(top_names)))
        ax.set_yticklabels(top_names, fontsize=9)
        ax.set_xlabel("Importance")
        ax.set_title(f"Top-{k} Prediction", fontsize=13)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(out_dir, "feature_importance_rf.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_feature_importance_gb(all_models: dict, out_dir: str = OUT_DIR):
    """
    Plot feature importance for the Gradient Boosting models.
    """
    if "gb" not in all_models:
        print("No Gradient Boosting models found, skipping.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    fig.suptitle("Gradient Boosting — Feature Importance (Top 20)", fontsize=16, fontweight="bold")

    for ax, (target, k) in zip(axes, TARGETS.items()):
        if target not in all_models["gb"]:
            continue

        model = all_models["gb"][target]
        clf = model.named_steps["clf"]
        feat_names = get_feature_names(model)
        importances = clf.feature_importances_

        indices = np.argsort(importances)[-20:]
        top_names = [feat_names[i] for i in indices]
        top_vals = importances[indices]

        ax.barh(range(len(top_names)), top_vals, color="#059669", edgecolor="white")
        ax.set_yticks(range(len(top_names)))
        ax.set_yticklabels(top_names, fontsize=9)
        ax.set_xlabel("Importance")
        ax.set_title(f"Top-{k} Prediction", fontsize=13)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(out_dir, "feature_importance_gb.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── 5. Model comparison bar chart ────────────────────────────────────

def plot_model_comparison(race_df: pd.DataFrame, out_dir: str = OUT_DIR):
    """
    Bar chart comparing average Top-K hit-rate across models.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    targets = race_df["Target"].unique()
    models = race_df["Model"].unique()
    x = np.arange(len(targets))
    width = 0.25
    colors = ["#6366f1", "#2563eb", "#059669"]

    for i, model_name in enumerate(models):
        subset = race_df[race_df["Model"] == model_name]
        vals = [subset[subset["Target"] == t]["Avg Hit-Rate"].values[0] for t in targets]
        bars = ax.bar(x + i * width, vals, width, label=model_name, color=colors[i % len(colors)],
                       edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Average Top-K Hit-Rate", fontsize=12)
    ax.set_title("Model Comparison — Average Race-Level Hit-Rate (2024 Test Set)", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(targets, fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    path = os.path.join(out_dir, "model_comparison_hitrate.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_row_level_comparison(row_df: pd.DataFrame, out_dir: str = OUT_DIR):
    """
    Grouped bar chart comparing F1-score across models.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    targets = row_df["Target"].unique()
    models = row_df["Model"].unique()
    x = np.arange(len(targets))
    width = 0.25
    colors = ["#6366f1", "#2563eb", "#059669"]

    for i, model_name in enumerate(models):
        subset = row_df[row_df["Model"] == model_name]
        vals = [subset[subset["Target"] == t]["F1"].values[0] for t in targets]
        bars = ax.bar(x + i * width, vals, width, label=model_name, color=colors[i % len(colors)],
                       edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("Model Comparison — F1 Score (2024 Test Set)", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(targets, fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    path = os.path.join(out_dir, "model_comparison_f1.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── 6. Per-race heatmap ─────────────────────────────────────────────

def plot_per_race_heatmap(per_race_df: pd.DataFrame, out_dir: str = OUT_DIR):
    """
    Heatmap showing RF hit-rate across all 2024 races.
    """
    if per_race_df.empty:
        return

    hr_cols = [c for c in per_race_df.columns if "Hit-Rate" in c]
    if not hr_cols:
        return

    data = per_race_df.set_index("EventName")[hr_cols].sort_index()

    fig, ax = plt.subplots(figsize=(8, max(6, len(data) * 0.35)))
    im = ax.imshow(data.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(hr_cols)))
    ax.set_xticklabels([c.replace(" Hit-Rate", "") for c in hr_cols], fontsize=11)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data.index, fontsize=9)

    for i in range(len(data)):
        for j in range(len(hr_cols)):
            val = data.values[i, j]
            color = "white" if val < 0.4 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=color)

    ax.set_title("Random Forest — Per-Race Hit-Rate (2024)", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.6, label="Hit-Rate")

    plt.tight_layout()
    path = os.path.join(out_dir, "per_race_heatmap.png")
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
    df_test = df[df["Year"] == 2024].copy()
    print(f"Test set: {len(df_test)} rows (2024 season)\n")

    os.makedirs(OUT_DIR, exist_ok=True)

    # Load models
    print("Loading models...")
    all_models = load_models("models")

    # 1. Row-level metrics
    print("\n1. Row-level classification metrics")
    print("=" * 60)
    row_df = row_level_comparison(all_models, df_test)
    print(row_df.to_string(index=False, float_format="{:.3f}".format))
    row_df.to_csv(os.path.join(OUT_DIR, "row_level_metrics.csv"), index=False)

    # 2. Race-level hit-rate
    print("\n\n2. Race-level Top-K hit-rate")
    print("=" * 60)
    race_df = race_level_comparison(all_models, df_test)
    print(race_df.to_string(index=False, float_format="{:.3f}".format))
    race_df.to_csv(os.path.join(OUT_DIR, "race_level_hitrate.csv"), index=False)

    # 3. Per-race breakdown
    print("\n\n3. Per-race breakdown (Random Forest)")
    print("=" * 60)
    per_race_df = per_race_breakdown(all_models, df_test, model_key="rf")
    print(per_race_df.to_string(index=False, float_format="{:.2f}".format))
    per_race_df.to_csv(os.path.join(OUT_DIR, "per_race_breakdown.csv"), index=False)

    # 4. Generate plots
    print("\n\n4. Generating plots...")
    print("-" * 40)
    plot_feature_importance_rf(all_models)
    plot_feature_importance_gb(all_models)
    plot_model_comparison(race_df)
    plot_row_level_comparison(row_df)
    plot_per_race_heatmap(per_race_df)

    print(f"\nAll results saved to {OUT_DIR}/")
    print("Done.")
