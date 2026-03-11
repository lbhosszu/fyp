"""
rolling_cv.py
=============
Rolling (expanding-window) cross-validation for F1 prediction models.

Instead of a single train/test split, this trains on progressively
larger windows and tests on the next unseen season:

  Fold 1: Train 2018-2020  →  Test 2021
  Fold 2: Train 2018-2021  →  Test 2022
  Fold 3: Train 2018-2022  →  Test 2023
  Fold 4: Train 2018-2023  →  Test 2024

This demonstrates that the model generalises across seasons and is
not just overfitting to a single test year. Results include per-fold
metrics and aggregated means with standard deviations.

All results are saved to the evaluation/ directory.
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score


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

CAT_FEATURES = ["TeamName", "EventName"]
NUM_FEATURES = [c for c in FEATURE_COLS if c not in CAT_FEATURES]

TARGETS = {"Top3": 3, "Top5": 5, "Top10": 10}

# Rolling folds: (train_end_year, test_year)
FOLDS = [
    (2020, 2021),
    (2021, 2022),
    (2022, 2023),
    (2023, 2024),
]

OUT_DIR = "evaluation"


# ── Pipeline builders ────────────────────────────────────────────────

def _preprocessor():
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
            ("num", "passthrough", NUM_FEATURES),
        ]
    )


def _preprocessor_scaled():
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
            ("num", StandardScaler(), NUM_FEATURES),
        ]
    )


MODEL_BUILDERS = {
    "Logistic Regression": lambda: Pipeline([
        ("prep", _preprocessor_scaled()),
        ("clf", LogisticRegression(
            max_iter=1000, class_weight="balanced",
            random_state=42, solver="lbfgs",
        )),
    ]),
    "Random Forest": lambda: Pipeline([
        ("prep", _preprocessor()),
        ("clf", RandomForestClassifier(
            n_estimators=400, max_depth=14,
            min_samples_split=6, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )),
    ]),
    "Gradient Boosting": lambda: Pipeline([
        ("prep", _preprocessor()),
        ("clf", GradientBoostingClassifier(
            n_estimators=300, max_depth=5,
            learning_rate=0.1, subsample=0.8, random_state=42,
        )),
    ]),
}


# ── Helpers ──────────────────────────────────────────────────────────

def topk_hit_rate(df_race, prob_col, k, truth_col):
    df_sorted = df_race.sort_values(prob_col, ascending=False)
    return df_sorted.head(k)[truth_col].mean()


def evaluate_fold(model_name, builder_fn, train_df, test_df, target, k):
    """Train and evaluate one model on one fold for one target."""
    X_train = train_df[FEATURE_COLS].copy()
    y_train = train_df[target].astype(int).values
    X_test = test_df[FEATURE_COLS].copy()
    y_test = test_df[target].astype(int).values

    pipe = builder_fn()
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


# ── Run rolling CV ───────────────────────────────────────────────────

def run_rolling_cv(df: pd.DataFrame) -> pd.DataFrame:
    """Run all models across all folds and targets."""
    rows = []

    for train_end, test_year in FOLDS:
        train_df = df[df["Year"] <= train_end].copy()
        test_df = df[df["Year"] == test_year].copy()
        print(f"\n  Fold: Train 2018-{train_end} ({len(train_df)} rows) → "
              f"Test {test_year} ({len(test_df)} rows)")

        for model_name, builder_fn in MODEL_BUILDERS.items():
            for target, k in TARGETS.items():
                f1, hit_rate = evaluate_fold(
                    model_name, builder_fn, train_df, test_df, target, k
                )
                rows.append({
                    "Model": model_name,
                    "Train": f"2018-{train_end}",
                    "Test Year": test_year,
                    "Target": f"Top-{k}",
                    "F1": f1,
                    "Hit-Rate": hit_rate,
                })
                print(f"    {model_name:25s} | Top-{k} | "
                      f"F1={f1:.3f} | Hit-Rate={hit_rate:.3f}")

    return pd.DataFrame(rows)


def summarise_cv(cv_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate results across folds: mean ± std for each model × target."""
    summary = (
        cv_df.groupby(["Model", "Target"])
        .agg(
            F1_mean=("F1", "mean"),
            F1_std=("F1", "std"),
            HitRate_mean=("Hit-Rate", "mean"),
            HitRate_std=("Hit-Rate", "std"),
            Num_Folds=("F1", "count"),
        )
        .reset_index()
    )

    # Create readable columns
    summary["F1 (mean ± std)"] = summary.apply(
        lambda r: f"{r['F1_mean']:.3f} ± {r['F1_std']:.3f}", axis=1
    )
    summary["Hit-Rate (mean ± std)"] = summary.apply(
        lambda r: f"{r['HitRate_mean']:.3f} ± {r['HitRate_std']:.3f}", axis=1
    )

    return summary


# ── Plotting ─────────────────────────────────────────────────────────

def plot_cv_by_year(cv_df: pd.DataFrame, out_dir: str = OUT_DIR):
    """
    Line chart: hit-rate across test years for each model,
    one subplot per target.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Rolling Cross-Validation — Hit-Rate Across Seasons",
                 fontsize=15, fontweight="bold")

    colors = {"Logistic Regression": "#6366f1",
              "Random Forest": "#2563eb",
              "Gradient Boosting": "#059669"}
    markers = {"Logistic Regression": "s",
               "Random Forest": "o",
               "Gradient Boosting": "D"}

    for ax, (target, k) in zip(axes, TARGETS.items()):
        target_label = f"Top-{k}"
        subset = cv_df[cv_df["Target"] == target_label]

        for model_name in MODEL_BUILDERS:
            model_data = subset[subset["Model"] == model_name].sort_values("Test Year")
            ax.plot(
                model_data["Test Year"], model_data["Hit-Rate"],
                marker=markers[model_name], markersize=8, linewidth=2,
                color=colors[model_name], label=model_name,
            )
            for _, row in model_data.iterrows():
                ax.annotate(
                    f"{row['Hit-Rate']:.2f}",
                    (row["Test Year"], row["Hit-Rate"]),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9, fontweight="bold",
                )

        ax.set_xlabel("Test Season", fontsize=11)
        ax.set_ylabel("Avg Hit-Rate", fontsize=11)
        ax.set_title(f"Top-{k} Prediction", fontsize=13)
        ax.set_xticks([2021, 2022, 2023, 2024])
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(out_dir, "rolling_cv_hitrate.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_cv_summary_bars(summary_df: pd.DataFrame, out_dir: str = OUT_DIR):
    """
    Bar chart with error bars showing mean ± std across all folds.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Rolling CV Summary — Mean ± Std Across 4 Folds",
                 fontsize=15, fontweight="bold")

    targets = summary_df["Target"].unique()
    models = list(MODEL_BUILDERS.keys())
    x = np.arange(len(targets))
    width = 0.25
    colors = ["#6366f1", "#2563eb", "#059669"]

    for ax, metric_mean, metric_std, ylabel in zip(
        axes,
        ["F1_mean", "HitRate_mean"],
        ["F1_std", "HitRate_std"],
        ["F1 Score", "Race-Level Hit-Rate"],
    ):
        for i, model_name in enumerate(models):
            subset = summary_df[summary_df["Model"] == model_name]
            means = [subset[subset["Target"] == t][metric_mean].values[0] for t in targets]
            stds = [subset[subset["Target"] == t][metric_std].values[0] for t in targets]

            bars = ax.bar(
                x + i * width, means, width, yerr=stds,
                label=model_name, color=colors[i],
                edgecolor="white", linewidth=0.5,
                capsize=4, error_kw={"linewidth": 1.5},
            )
            for bar, val in zip(bars, means):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.03,
                    f"{val:.2f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold",
                )

        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xticks(x + width)
        ax.set_xticklabels(targets, fontsize=12)
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(out_dir, "rolling_cv_summary.png")
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
    print(f"Loaded {len(df)} rows\n")

    os.makedirs(OUT_DIR, exist_ok=True)

    # Run rolling CV
    print("=" * 60)
    print("ROLLING CROSS-VALIDATION")
    print("=" * 60)
    cv_df = run_rolling_cv(df)
    cv_df.to_csv(os.path.join(OUT_DIR, "rolling_cv_results.csv"), index=False)

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY (mean ± std across folds)")
    print("=" * 60)
    summary = summarise_cv(cv_df)
    print(summary[["Model", "Target", "F1 (mean ± std)", "Hit-Rate (mean ± std)"]].to_string(index=False))
    summary.to_csv(os.path.join(OUT_DIR, "rolling_cv_summary.csv"), index=False)

    # Plots
    print(f"\n{'=' * 60}")
    print("Generating plots...")
    print("=" * 60)
    plot_cv_by_year(cv_df)
    plot_cv_summary_bars(summary)

    print(f"\nAll CV results saved to {OUT_DIR}/")
    print("Done.")
