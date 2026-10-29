"""
All tunables in one place. Change values here, not in the other files.
"""

# ── Dataset ──────────────────────────────────────────────────────────
DATASET = "sh0416/wildchat-1m-tagged"
MIN_CLASS_FRAC = 0.02          # merge classes below 2 % into "Other"

# ── Deduplication ────────────────────────────────────────────────────
DEDUPE_PREFIX_LEN = 200        # first N chars used as the dedup key
DEDUPE_MAX_PER_WEEK = 5        # keep at most this many copies per key per week

# ── Weekly batches ───────────────────────────────────────────────────
POOL_PER_WEEK = 3000           # max chats kept in pool.parquet per week
SAMPLE_PER_WEEK = 1000         # chats drawn from the pool per seed
MIN_WEEK_SIZE = 500            # drop weeks smaller than this

# ── Train / test split ───────────────────────────────────────────────
INITIAL_WEEKS = 12             # first N weeks = initial training set
WINDOW_SIZE = 12               # sliding training window (last N weeks)

# ── TF-IDF + LogisticRegression ──────────────────────────────────────
TFIDF_MAX_FEATURES = 50_000
TFIDF_NGRAM = (1, 2)
TFIDF_MIN_DF = 3
LR_MAX_ITER = 1000

# ── Seeds for multi-run experiment ───────────────────────────────────
SEEDS = [42, 123, 256, 512, 1024]

# ── Drift detector ──────────────────────────────────────────────────
DRIFT_CALIBRATION_WEEKS = 6    # first N test weeks used to set thresholds
DRIFT_SIGMAS = [2, 3, 4]       # run rule-4 at each of these sigma levels
DRIFT_COOLDOWN_WEEKS = 2       # min gap between consecutive retrains

# ── Score-triggered rule (rule 5) ────────────────────────────────────
SCORE_REFERENCE_WEEKS = 6      # first N test weeks used to compute reference F1
SCORE_CHECK_WINDOW = 2         # average of last N weeks for the check
SCORE_DROP_THRESHOLD = 0.05    # default threshold: retrain when recent F1 < reference - this
SCORE_THRESHOLDS = [0.02, 0.03, 0.05]  # score-triggered threshold variants
SCORE_COOLDOWN_WEEKS = 2       # cooldown between retrains (same as drift)
LOW_F1_THRESHOLD = 0.42        # threshold for "weeks below 0.42 F1" metric

# ── MLflow ───────────────────────────────────────────────────────────
MLFLOW_EXPERIMENT = "retrain-strategies"

