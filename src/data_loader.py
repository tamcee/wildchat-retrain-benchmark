"""
data_loader.py — build pool.parquet once, then sample per seed.

Pool:   HF dataset → English filter → last user message → drop Unknown →
        merge rare classes → weekly batches → dedupe → cap at ~3000/week.
Sample: draw min(1000, group_size) from pool for a given seed.
"""
import os
import pandas as pd
from datasets import load_dataset

import config as cfg


# ── helpers (reused from drift_effect.py) ────────────────────────────

def last_user_prompt(conversation):
    """Return the text of the last user turn in the conversation list."""
    for turn in reversed(conversation):
        if turn["role"] == "user":
            return turn["content"] or ""
    return ""


def _assign_week(ts_series):
    """Turn a datetime series into 'YYYY-Www' ISO week strings."""
    iso = ts_series.dt.isocalendar()
    return iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)


# ── pool builder ─────────────────────────────────────────────────────

def build_pool(pool_path="data/pool.parquet", force=False):
    """
    Download the HF dataset once, clean it, and save a pool parquet.
    Returns the pool DataFrame.
    """
    if os.path.exists(pool_path) and not force:
        print(f"Pool already exists at {pool_path}, loading it.")
        return pd.read_parquet(pool_path)

    print("Loading HF dataset (this takes a while the first time)...")
    ds = load_dataset(cfg.DATASET, split="train")

    # step 1: read metadata columns first (fast)
    meta = ds.select_columns(["timestamp", "language", "category"]).to_pandas()
    meta["week_id"] = _assign_week(meta["timestamp"])

    # step 2: keep only English, non-Unknown, non-null category
    mask = (
        (meta["language"] == "English")
        & meta["category"].notna()
        & (meta["category"] != "Unknown")
    )
    meta = meta[mask].copy()
    print(f"English non-Unknown rows: {len(meta):,}")

    # step 3: extract the last user prompt (needs conversation column)
    indices = meta.index.tolist()
    convs = ds.select(indices).select_columns(["conversation"])
    meta["text"] = [last_user_prompt(r["conversation"]) for r in convs]
    meta = meta[meta["text"].str.len() > 0].copy()

    # step 4: merge rare classes into "Other"
    share = meta["category"].value_counts(normalize=True)
    keep_classes = share[share >= cfg.MIN_CLASS_FRAC].index
    meta["label"] = meta["category"].where(
        meta["category"].isin(keep_classes), "Other"
    )
    print("Classes kept:", sorted(meta["label"].unique()))

    # step 5: dedupe — shuffle, then keep at most N copies of same prefix per week
    meta = meta.sample(frac=1, random_state=0).reset_index(drop=True)  # shuffle
    prefix = meta["text"].str[:cfg.DEDUPE_PREFIX_LEN].str.lower().str.strip()
    meta["_prefix"] = prefix
    meta = (
        meta.groupby(["week_id", "_prefix"], sort=False)
        .head(cfg.DEDUPE_MAX_PER_WEEK)
        .drop(columns=["_prefix"])
        .reset_index(drop=True)
    )
    print(f"After dedupe: {len(meta):,} rows")

    # step 6: cap each week at POOL_PER_WEEK
    meta = (
        meta.groupby("week_id", sort=False)
        .head(cfg.POOL_PER_WEEK)
        .reset_index(drop=True)
    )

    # step 7: drop weeks below minimum
    week_counts = meta["week_id"].value_counts()
    good_weeks = week_counts[week_counts >= cfg.MIN_WEEK_SIZE].index
    meta = meta[meta["week_id"].isin(good_weeks)].copy()
    print(f"Weeks kept: {meta['week_id'].nunique()} "
          f"(dropped {len(week_counts) - len(good_weeks)} small weeks)")

    # keep only needed columns, sorted by timestamp
    pool = (
        meta[["week_id", "text", "label", "timestamp"]]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    os.makedirs(os.path.dirname(pool_path), exist_ok=True)
    pool.to_parquet(pool_path, index=False)
    print(f"Saved pool to {pool_path} ({len(pool):,} rows)")
    return pool


# ── per-seed sampler ─────────────────────────────────────────────────

def sample_from_pool(pool, seed, sample_dir="data"):
    """
    Draw min(SAMPLE_PER_WEEK, group_size) rows per week from the pool.
    Saves to data/sample_seed{seed}.parquet and returns the DataFrame.
    """
    path = os.path.join(sample_dir, f"sample_seed{seed}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)

    parts = []
    for week_id, group in pool.groupby("week_id", sort=False):
        n = min(cfg.SAMPLE_PER_WEEK, len(group))
        parts.append(group.sample(n=n, random_state=seed))
    sample = pd.concat(parts).sort_values("timestamp").reset_index(drop=True)

    os.makedirs(sample_dir, exist_ok=True)
    sample.to_parquet(path, index=False)
    print(f"Saved sample (seed={seed}) to {path} ({len(sample):,} rows)")
    return sample


# ── tiny-sample generator for CI smoke tests ─────────────────────────

def make_tiny_sample(pool, out_path="data/tiny_sample.parquet",
                     weeks=20, chats_per_week=200, seed=99):
    """
    Build a small parquet for CI: ~20 weeks × 200 chats.
    This file is committed to the repo so CI never touches HF.
    """
    all_weeks = sorted(pool["week_id"].unique())
    chosen = all_weeks[:weeks]
    parts = []
    for w in chosen:
        g = pool[pool["week_id"] == w]
        n = min(chats_per_week, len(g))
        parts.append(g.sample(n=n, random_state=seed))
    tiny = pd.concat(parts).sort_values("timestamp").reset_index(drop=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tiny.to_parquet(out_path, index=False)
    print(f"Saved tiny sample to {out_path} ({len(tiny):,} rows)")
    return tiny


if __name__ == "__main__":
    pool = build_pool()
    for s in cfg.SEEDS:
        sample_from_pool(pool, s)
    make_tiny_sample(pool)
