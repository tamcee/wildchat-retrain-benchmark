"""
viz.py — plots and summary tables.

Outputs:
  results/f1_over_time.png   — macro-F1 per week for every rule
  results/drift_scores.png   — OOV share and JS distance over time
  results/summary.csv        — per-rule aggregate table
  results/results.csv        — full per-week data
"""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import config as cfg


def save_results_csv(all_results, path="results/results.csv"):
    """Save the full per-week DataFrame."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    all_results.to_csv(path, index=False)
    print(f"Saved {path}")


def build_summary(all_results, rules_meta):
    """
    Build per-rule summary: avg macro-F1 ± std, weeks below 0.42 F1, retrains, time.
    rules_meta: dict rule_name → needs_labels (bool).

    Aggregates across seeds.
    """
    rows = []
    low_f1_th = getattr(cfg, "LOW_F1_THRESHOLD", 0.42)
    for rule_name in all_results["rule"].unique():
        subset = all_results[all_results["rule"] == rule_name]
        # per-seed stats
        seed_stats = []
        for seed in subset["seed"].unique():
            s = subset[subset["seed"] == seed]
            seed_stats.append({
                "mean_f1": s["macro_f1"].mean(),
                "weeks_below_0.42": (s["macro_f1"] < low_f1_th).sum(),
                "n_retrains": s["retrained"].sum(),
                "total_train_time": s["train_time_s"].sum(),
            })
        ss = pd.DataFrame(seed_stats)
        needs = rules_meta.get(rule_name, False)
        n_seeds = len(seed_stats)

        def _fmt(mean_val, std_val, decimals=4):
            if n_seeds > 1 and not np.isnan(std_val):
                return f"{mean_val:.{decimals}f} ± {std_val:.{decimals}f}"
            return f"{mean_val:.{decimals}f}"

        rows.append({
            "rule": rule_name,
            "avg_macro_f1": _fmt(ss['mean_f1'].mean(), ss['mean_f1'].std(), 4),
            "weeks_below_0.42_f1": _fmt(ss['weeks_below_0.42'].mean(), ss['weeks_below_0.42'].std(), 1),
            "avg_retrains": _fmt(ss['n_retrains'].mean(), ss['n_retrains'].std(), 1),
            "avg_train_time_s": _fmt(ss['total_train_time'].mean(), ss['total_train_time'].std(), 1),
            "needs_labels": needs,
        })
    return pd.DataFrame(rows)


def save_summary(summary, path="results/summary.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    summary.to_csv(path, index=False)
    print(f"Saved {path}")


def plot_f1_over_time(all_results, path="results/f1_over_time.png"):
    """
    One line per rule (mean across seeds), shaded ±1 std.
    Diamond markers where retraining happened.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 6))

    rules = sorted(all_results["rule"].unique())
    weeks = sorted(all_results["week_id"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(rules)))

    for rule, color in zip(rules, colors):
        subset = all_results[all_results["rule"] == rule]
        # aggregate across seeds per week
        agg = subset.groupby("week_id").agg(
            f1_mean=("macro_f1", "mean"),
            f1_std=("macro_f1", "std"),
            retrain_frac=("retrained", "mean"),
        ).reindex(weeks)
        agg["f1_std"] = agg["f1_std"].fillna(0)

        x = range(len(weeks))
        ax.plot(x, agg["f1_mean"], label=rule, color=color, linewidth=1.5)
        ax.fill_between(
            x,
            agg["f1_mean"] - agg["f1_std"],
            agg["f1_mean"] + agg["f1_std"],
            alpha=0.15,
            color=color,
        )
        # mark retraining points (where >50% of seeds retrained)
        retrain_mask = agg["retrain_frac"] > 0.5
        retrain_x = [i for i, m in enumerate(retrain_mask) if m]
        retrain_y = [agg["f1_mean"].iloc[i] for i in retrain_x]
        ax.scatter(retrain_x, retrain_y, marker="D", color=color, s=40, zorder=5)

    ax.set_xticks(range(0, len(weeks), max(1, len(weeks) // 15)))
    ax.set_xticklabels(
        [weeks[i] for i in range(0, len(weeks), max(1, len(weeks) // 15))],
        rotation=45, ha="right", fontsize=8,
    )
    ax.set_xlabel("Week")
    ax.set_ylabel("Macro F1")
    ax.set_title("Macro F1 per week by retraining rule (mean ± 1 std across seeds)")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_drift_scores(all_results, thresholds_by_sigma, path="results/drift_scores.png"):
    """
    Two subplots: OOV share and JS distance over time.
    Horizontal dashed lines at thresholds for each sigma level.
    Uses only seed=first seed for clarity.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # use the "never" rule (raw scores, no retraining effect) and mean across seeds
    base = all_results[all_results["rule"] == "never"]
    weeks = sorted(base["week_id"].unique())
    agg = base.groupby("week_id").agg(
        oov_mean=("oov_share", "mean"),
        oov_std=("oov_share", "std"),
        js_mean=("js_distance", "mean"),
        js_std=("js_distance", "std"),
    ).reindex(weeks)
    agg["oov_std"] = agg["oov_std"].fillna(0)
    agg["js_std"] = agg["js_std"].fillna(0)

    x = range(len(weeks))

    # OOV share
    ax1.plot(x, agg["oov_mean"], color="steelblue", label="OOV share")
    ax1.fill_between(x, agg["oov_mean"] - agg["oov_std"],
                     agg["oov_mean"] + agg["oov_std"], alpha=0.2, color="steelblue")
    for sigma, th in thresholds_by_sigma.items():
        ax1.axhline(th["oov_threshold"], linestyle="--", linewidth=0.8,
                     label=f"threshold ({sigma}σ)")
    ax1.set_ylabel("OOV share")
    ax1.set_title("Drift scores over time (mean across seeds, 'never' rule)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # JS distance
    ax2.plot(x, agg["js_mean"], color="darkorange", label="JS distance")
    ax2.fill_between(x, agg["js_mean"] - agg["js_std"],
                     agg["js_mean"] + agg["js_std"], alpha=0.2, color="darkorange")
    for sigma, th in thresholds_by_sigma.items():
        ax2.axhline(th["js_threshold"], linestyle="--", linewidth=0.8,
                     label=f"threshold ({sigma}σ)")
    ax2.set_ylabel("JS distance")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    ax2.set_xticks(range(0, len(weeks), max(1, len(weeks) // 15)))
    ax2.set_xticklabels(
        [weeks[i] for i in range(0, len(weeks), max(1, len(weeks) // 15))],
        rotation=45, ha="right", fontsize=8,
    )
    ax2.set_xlabel("Week")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_tradeoff(all_results, rules_meta, path="results/tradeoff.png"):
    """
    Scatter of average macro F1 (y) vs average number of retrains (x).
    One labeled point per rule, with error bars on F1 (mean ± std across seeds).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    points = []
    for rule_name in all_results["rule"].unique():
        subset = all_results[all_results["rule"] == rule_name]
        seed_stats = []
        for seed in subset["seed"].unique():
            s = subset[subset["seed"] == seed]
            seed_stats.append({
                "mean_f1": s["macro_f1"].mean(),
                "n_retrains": s["retrained"].sum(),
            })
        ss = pd.DataFrame(seed_stats)
        points.append({
            "rule": rule_name,
            "f1_mean": ss["mean_f1"].mean(),
            "f1_std": ss["mean_f1"].std() if len(ss) > 1 and not np.isnan(ss["mean_f1"].std()) else 0.0,
            "retrains_mean": ss["n_retrains"].mean(),
            "needs_labels": rules_meta.get(rule_name, False),
        })

    pdf = pd.DataFrame(points)

    colors = []
    markers = []
    for _, r in pdf.iterrows():
        name = r["rule"]
        if r["needs_labels"]:
            colors.append("crimson")
            markers.append("^")
        elif "drift" in name:
            colors.append("teal")
            markers.append("s")
        elif name == "never":
            colors.append("dimgray")
            markers.append("X")
        else:
            colors.append("royalblue")
            markers.append("o")

    for i, row in pdf.iterrows():
        ax.errorbar(
            row["retrains_mean"],
            row["f1_mean"],
            yerr=row["f1_std"],
            fmt=markers[i],
            color=colors[i],
            ecolor=colors[i],
            elinewidth=1.5,
            capsize=4,
            capthick=1.5,
            markersize=9,
            alpha=0.9,
            zorder=4,
        )

        # Smart label offsets to prevent collision
        name = row["rule"]
        offset_x = 0.8
        offset_y = 0.0012

        if name == "drift_2σ":
            offset_x = -2.2
            offset_y = 0.0018
        elif name == "drift_3σ":
            offset_x = 0.8
            offset_y = -0.0022
        elif name == "drift_4σ":
            offset_x = -2.5
            offset_y = 0.0018
        elif name == "score_0.02":
            offset_x = 0.8
            offset_y = 0.0015
        elif name == "score_0.03":
            offset_x = 0.8
            offset_y = -0.0020
        elif name == "score_0.05":
            offset_x = 0.8
            offset_y = -0.0025
        elif name == "every_4_weeks":
            offset_x = 0.8
            offset_y = 0.0018
        elif name == "never":
            offset_x = 0.8
            offset_y = -0.0015
        elif name == "every_week":
            offset_x = -6.5
            offset_y = -0.0015

        ax.annotate(
            name,
            xy=(row["retrains_mean"], row["f1_mean"]),
            xytext=(row["retrains_mean"] + offset_x, row["f1_mean"] + offset_y),
            fontsize=9,
            fontweight="bold" if "every_4_weeks" in name or "drift" in name else "normal",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75),
            zorder=5,
        )

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="Scheduled (No labels)",
               markerfacecolor="royalblue", markersize=8),
        Line2D([0], [0], marker="s", color="w", label="Drift-triggered (No labels)",
               markerfacecolor="teal", markersize=8),
        Line2D([0], [0], marker="^", color="w", label="Score-triggered (Needs labels)",
               markerfacecolor="crimson", markersize=8),
        Line2D([0], [0], marker="X", color="w", label="Baseline (Never retrain)",
               markerfacecolor="dimgray", markersize=8),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.9, fontsize=9)

    ax.set_xlabel("Average Number of Retrains (across 45 test weeks)", fontsize=10)
    ax.set_ylabel("Average Macro F1 (mean ± 1 std across seeds)", fontsize=10)
    ax.set_title("Performance vs. Retraining Cost Tradeoff by Strategy", fontsize=12, pad=10)
    ax.grid(True, linestyle="--", alpha=0.4)

    x_max = max(pdf["retrains_mean"].max() + 5, 50)
    ax.set_xlim(-2, x_max)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")
