"""
Prepares structured JSON data for the web dashboard.
"""
import os
import json
import pandas as pd
import numpy as np


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


def prepare():
    results = pd.read_csv("results/results.csv")
    summary = pd.read_csv("results/summary.csv")

    summary_list = summary.to_dict(orient="records")

    weeks = sorted(results["week_id"].unique().tolist())
    rules = sorted(results["rule"].unique().tolist())
    seeds = sorted(results["seed"].unique().tolist())

    weekly_agg = {}
    for rule in rules:
        rule_df = results[results["rule"] == rule]
        agg = rule_df.groupby("week_id").agg(
            f1_mean=("macro_f1", "mean"),
            f1_std=("macro_f1", "std"),
            f1_min=("macro_f1", "min"),
            f1_max=("macro_f1", "max"),
            acc_mean=("accuracy", "mean"),
            retrain_count=("retrained", "sum"),
            retrain_frac=("retrained", "mean"),
            oov_mean=("oov_share", "mean"),
            js_mean=("js_distance", "mean")
        ).reindex(weeks).fillna(0)

        weekly_agg[rule] = {
            "weeks": weeks,
            "f1_mean": [round(float(x), 4) for x in agg["f1_mean"]],
            "f1_std": [round(float(x), 4) for x in agg["f1_std"]],
            "f1_min": [round(float(x), 4) for x in agg["f1_min"]],
            "f1_max": [round(float(x), 4) for x in agg["f1_max"]],
            "acc_mean": [round(float(x), 4) for x in agg["acc_mean"]],
            "retrain_count": [int(x) for x in agg["retrain_count"]],
            "retrain_frac": [round(float(x), 2) for x in agg["retrain_frac"]],
            "retrain_weeks": [w for w, r in zip(weeks, agg["retrain_frac"]) if r > 0.5]
        }

    # Per-seed trajectories
    per_seed = {}
    for seed in seeds:
        per_seed[str(seed)] = {}
        seed_df = results[results["seed"] == seed]
        for rule in rules:
            s_r = seed_df[seed_df["rule"] == rule].sort_values("week_id")
            per_seed[str(seed)][rule] = {
                "f1": [round(float(x), 4) for x in s_r["macro_f1"]],
                "retrained": [bool(x) for x in s_r["retrained"]],
                "oov": [round(float(x), 4) for x in s_r["oov_share"]],
                "js": [round(float(x), 4) for x in s_r["js_distance"]]
            }

    # Drift scores from "never" rule (baseline)
    never_df = results[results["rule"] == "never"]
    drift_agg = never_df.groupby("week_id").agg(
        oov_mean=("oov_share", "mean"),
        oov_std=("oov_share", "std"),
        js_mean=("js_distance", "mean"),
        js_std=("js_distance", "std")
    ).reindex(weeks).fillna(0)

    cal_weeks = weeks[:6]
    cal_never = never_df[never_df["week_id"].isin(cal_weeks)]
    oov_cal_mean = float(cal_never["oov_share"].mean())
    oov_cal_std = float(cal_never["oov_share"].std())
    js_cal_mean = float(cal_never["js_distance"].mean())
    js_cal_std = float(cal_never["js_distance"].std())

    drift_data = {
        "weeks": weeks,
        "cal_weeks": cal_weeks,
        "oov_mean": [round(float(x), 4) for x in drift_agg["oov_mean"]],
        "oov_std": [round(float(x), 4) for x in drift_agg["oov_std"]],
        "js_mean": [round(float(x), 4) for x in drift_agg["js_mean"]],
        "js_std": [round(float(x), 4) for x in drift_agg["js_std"]],
        "thresholds": {
            "2sigma": {
                "oov": round(oov_cal_mean + 2 * oov_cal_std, 4),
                "js": round(js_cal_mean + 2 * js_cal_std, 4)
            },
            "3sigma": {
                "oov": round(oov_cal_mean + 3 * oov_cal_std, 4),
                "js": round(js_cal_mean + 3 * js_cal_std, 4)
            },
            "4sigma": {
                "oov": round(oov_cal_mean + 4 * oov_cal_std, 4),
                "js": round(js_cal_mean + 4 * js_cal_std, 4)
            },
            "cal_mean": {
                "oov": round(oov_cal_mean, 4),
                "js": round(js_cal_mean, 4)
            }
        }
    }

    tradeoff_pts = []
    for item in summary_list:
        r = item["rule"]
        f1_str = str(item["avg_macro_f1"]).split("±")[0].strip()
        f1_std_str = str(item["avg_macro_f1"]).split("±")[1].strip() if "±" in str(item["avg_macro_f1"]) else "0"
        ret_str = str(item["avg_retrains"]).split("±")[0].strip()
        ret_std_str = str(item["avg_retrains"]).split("±")[1].strip() if "±" in str(item["avg_retrains"]) else "0"
        below_str = str(item["weeks_below_0.42_f1"]).split("±")[0].strip()
        time_str = str(item["avg_train_time_s"]).split("±")[0].strip()

        # Category
        if r in ["every_week", "every_4_weeks"]:
            cat = "Scheduled"
            color = "#6366f1"  # indigo
        elif "drift" in r:
            cat = "Drift-Triggered"
            color = "#10b981"  # emerald
        elif "score" in r:
            cat = "Score-Triggered"
            color = "#f43f5e"  # rose
        else:
            cat = "Baseline"
            color = "#71717a"  # zinc

        tradeoff_pts.append({
            "rule": r,
            "category": cat,
            "color": color,
            "f1_mean": float(f1_str),
            "f1_std": float(f1_std_str),
            "retrains_mean": float(ret_str),
            "retrains_std": float(ret_std_str),
            "weeks_below_042": float(below_str),
            "train_time_s": float(time_str),
            "needs_labels": bool(item["needs_labels"])
        })

    os.makedirs("dashboard/web", exist_ok=True)
    payload = {
        "summary": summary_list,
        "tradeoff": tradeoff_pts,
        "weekly": weekly_agg,
        "per_seed": per_seed,
        "drift": drift_data,
        "weeks": weeks,
        "rules": rules,
        "seeds": seeds,
        "raw_records": [
            {
                "week_id": row["week_id"],
                "macro_f1": round(float(row["macro_f1"]), 4),
                "accuracy": round(float(row["accuracy"]), 4),
                "oov_share": round(float(row["oov_share"]), 4),
                "js_distance": round(float(row["js_distance"]), 4),
                "retrained": bool(row["retrained"]),
                "train_time_s": round(float(row["train_time_s"]), 2),
                "rule": row["rule"],
                "seed": int(row["seed"])
            }
            for _, row in results.iterrows()
        ]
    }

    target_path = "dashboard/web/data.json"
    with open(target_path, "w") as f:
        json.dump(payload, f, cls=NpEncoder)

    js_target_path = "dashboard/web/data.js"
    with open(js_target_path, "w") as f:
        f.write("window.DASHBOARD_DATA = " + json.dumps(payload, cls=NpEncoder) + ";\n")

    print(f"Generated {target_path} and {js_target_path} successfully ({os.path.getsize(target_path) / 1024:.1f} KB)")


if __name__ == "__main__":
    prepare()
