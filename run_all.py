"""
run_all.py — entry point: runs 5 seeds × 7 rule variants, logs to MLflow.

Rule variants:
  never, every_week, every_4_weeks,
  drift_triggered@2σ, drift_triggered@3σ, drift_triggered@4σ,
  score_triggered

Usage:
  python run_all.py                     # full run (5 seeds)
  python run_all.py --seeds 1           # quick test (1 seed)
  python run_all.py --max-weeks 20      # limit weeks
  python run_all.py --pool data/tiny_sample.parquet   # CI mode
"""
import argparse
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import numpy as np
import pandas as pd
import mlflow

# make sure the project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

import config as cfg
from src.data_loader import build_pool, sample_from_pool
from src.experiment import run_experiment
from src.drift import calibrate_thresholds
from src.rules import RULES
from src.viz import (
    save_results_csv,
    build_summary,
    save_summary,
    plot_f1_over_time,
    plot_drift_scores,
    plot_tradeoff,
)


def parse_args():
    p = argparse.ArgumentParser(description="Retraining strategy comparison")
    p.add_argument("--seeds", type=int, default=len(cfg.SEEDS),
                   help="Number of seeds to use (default: all 5)")
    p.add_argument("--max-weeks", type=int, default=None,
                   help="Cap total weeks (train + test) for faster runs")
    p.add_argument("--pool", type=str, default="data/pool.parquet",
                   help="Path to pool parquet (use data/tiny_sample.parquet for CI)")
    p.add_argument("--sample-dir", type=str, default="data",
                   help="Directory for per-seed sample parquets")
    p.add_argument("--results-dir", type=str, default="results",
                   help="Directory for output files")
    return p.parse_args()


def _trim_weeks(df, max_weeks):
    """Keep only the first max_weeks weeks."""
    if max_weeks is None:
        return df
    weeks = sorted(df["week_id"].unique())[:max_weeks]
    return df[df["week_id"].isin(weeks)].copy()


def main():
    args = parse_args()
    seeds = cfg.SEEDS[: args.seeds]

    # ── load or build pool ───────────────────────────────────────
    if args.pool == "data/pool.parquet":
        pool = build_pool(args.pool)
    else:
        # CI mode: pool is a tiny committed parquet
        pool = pd.read_parquet(args.pool)
        print(f"Loaded pool from {args.pool} ({len(pool):,} rows)")

    # ── define rule variants ─────────────────────────────────────
    # "drift_triggered" gets expanded into one variant per sigma
    # "score_triggered" gets expanded into one variant per threshold (0.02, 0.03, 0.05)
    rule_variants = []
    for name in RULES:
        if name == "drift_triggered":
            for sigma in cfg.DRIFT_SIGMAS:
                rule_variants.append((f"drift_{sigma}σ", name, sigma, None))
        elif name == "score_triggered":
            for th in cfg.SCORE_THRESHOLDS:
                rule_variants.append((f"score_{th}", name, 3.0, th))
        else:
            rule_variants.append((name, name, None, None))

    # ── MLflow setup ─────────────────────────────────────────────
    # MLflow 3.x requires a database backend; use a local SQLite file
    project_root = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(project_root, "mlruns.db")
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment(cfg.MLFLOW_EXPERIMENT)

    all_results = []
    # collect thresholds for the drift-score plot (one per sigma per seed)
    thresholds_collector = {s: [] for s in cfg.DRIFT_SIGMAS}

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")

        sample = sample_from_pool(pool, seed, args.sample_dir)
        sample = _trim_weeks(sample, args.max_weeks)
        n_weeks = sample["week_id"].nunique()
        print(f"  {len(sample):,} rows, {n_weeks} weeks")

        for display_name, rule_name, sigma, score_th in rule_variants:
            print(f"\n  ── rule: {display_name} ──")
            sigma_val = sigma if sigma is not None else 3.0
            df = run_experiment(rule_name, sample, sigma=sigma_val, score_threshold=score_th)
            df["rule"] = display_name
            df["seed"] = seed
            all_results.append(df)

            # collect thresholds for drift_triggered runs
            if rule_name == "drift_triggered" and sigma is not None:
                from src.drift import calibrate_thresholds as _cal
                oov_hist = df["oov_share"].tolist()
                js_hist = df["js_distance"].tolist()
                if len(oov_hist) >= cfg.DRIFT_CALIBRATION_WEEKS:
                    th = _cal(oov_hist, js_hist, sigma)
                    thresholds_collector[sigma].append(th)

            # log to MLflow
            with mlflow.start_run(run_name=f"{display_name}_seed{seed}"):
                mlflow.log_params({
                    "rule": display_name,
                    "seed": seed,
                    "window_size": cfg.WINDOW_SIZE,
                    "initial_weeks": cfg.INITIAL_WEEKS,
                    "sigma": sigma_val,
                    "score_threshold": score_th if score_th is not None else "N/A",
                    "n_weeks": n_weeks,
                })
                low_f1_th = getattr(cfg, "LOW_F1_THRESHOLD", 0.42)
                mlflow.log_metrics({
                    "mean_macro_f1": df["macro_f1"].mean(),
                    "worst_macro_f1": df["macro_f1"].min(),
                    "weeks_below_0.42": int((df["macro_f1"] < low_f1_th).sum()),
                    "n_retrains": int(df["retrained"].sum()),
                    "total_train_time_s": df["train_time_s"].sum(),
                })

            n_ret = int(df["retrained"].sum())
            n_below = int((df["macro_f1"] < getattr(cfg, "LOW_F1_THRESHOLD", 0.42)).sum())
            print(f"    mean F1={df['macro_f1'].mean():.3f}  "
                  f"below_0.42={n_below}  "
                  f"retrains={n_ret}  "
                  f"time={df['train_time_s'].sum():.1f}s")

    # ── aggregate ────────────────────────────────────────────────
    all_df = pd.concat(all_results, ignore_index=True)

    # build needs_labels lookup
    rules_meta = {}
    for display_name, rule_name, sigma, score_th in rule_variants:
        _, needs = RULES[rule_name]
        rules_meta[display_name] = needs

    # ── save outputs ─────────────────────────────────────────────
    os.makedirs(args.results_dir, exist_ok=True)
    save_results_csv(all_df, os.path.join(args.results_dir, "results.csv"))

    summary = build_summary(all_df, rules_meta)
    save_summary(summary, os.path.join(args.results_dir, "summary.csv"))
    print("\n" + summary.to_string(index=False))

    plot_f1_over_time(all_df, os.path.join(args.results_dir, "f1_over_time.png"))
    plot_tradeoff(all_df, rules_meta, os.path.join(args.results_dir, "tradeoff.png"))

    # average thresholds across seeds for the plot
    avg_thresholds = {}
    for sigma, th_list in thresholds_collector.items():
        if th_list:
            avg_thresholds[sigma] = {
                "oov_threshold": np.mean([t["oov_threshold"] for t in th_list]),
                "js_threshold": np.mean([t["js_threshold"] for t in th_list]),
            }
    if avg_thresholds:
        plot_drift_scores(all_df, avg_thresholds,
                          os.path.join(args.results_dir, "drift_scores.png"))

    print("\nDone. Results in", args.results_dir)


if __name__ == "__main__":
    main()

