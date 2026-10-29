"""
experiment.py — run one (rule, seed) end-to-end.

Flow for each test week:
  1. Predict on the week's texts (model has NOT seen true labels yet).
  2. Record scores and drift signals.
  3. True labels "arrive" (simulated).
  4. The rule decides whether to retrain.
  5. If retraining: refit on the last WINDOW_SIZE weeks.
"""
import numpy as np
import pandas as pd

import config as cfg
from src.model import fit_pipeline, predict, score, get_vocab, get_classes
from src.drift import oov_share, js_distance, calibrate_thresholds, is_drift
from src.rules import RULES


def run_experiment(rule_name, sample_df, sigma=3.0, score_threshold=None):
    """
    Run the full streaming experiment for one rule on one sample.

    Parameters
    ----------
    rule_name : str
        Key in RULES registry (e.g. "never", "drift_triggered").
    sample_df : pd.DataFrame
        Columns: week_id, text, label, timestamp.
    sigma : float
        Sigma level for drift thresholds (used only by drift_triggered).
    score_threshold : float, optional
        F1 drop threshold for score_triggered rule.

    Returns
    -------
    pd.DataFrame with one row per test week.
    """
    rule_fn, needs_labels = RULES[rule_name]

    weeks = sorted(sample_df["week_id"].unique())
    train_weeks = weeks[: cfg.INITIAL_WEEKS]
    test_weeks = weeks[cfg.INITIAL_WEEKS :]

    if len(test_weeks) == 0:
        raise ValueError("No test weeks available — need more data.")

    # ── initial training ─────────────────────────────────────────────
    train_df = sample_df[sample_df["week_id"].isin(train_weeks)]
    pipe, init_train_time = fit_pipeline(
        train_df["text"].tolist(), train_df["label"].tolist()
    )

    # predictions on the training set (for JS baseline)
    train_pred_labels = predict(pipe, train_df["text"].tolist())
    all_classes = sorted(sample_df["label"].unique())

    # ── bookkeeping ──────────────────────────────────────────────────
    current_window_start = 0  # index into `weeks`
    oov_history = []
    js_history = []
    f1_history = []          # true macro-F1 per test week (for rule 5)
    thresholds = None        # calibrated after DRIFT_CALIBRATION_WEEKS
    last_retrain_idx = None  # test_week_index of last retrain

    results = []

    for i, wk in enumerate(test_weeks):
        week_df = sample_df[sample_df["week_id"] == wk]
        texts = week_df["text"].tolist()
        true_labels = week_df["label"].tolist()

        # ── step 1: predict (before seeing true labels) ──────────
        preds = predict(pipe, texts)
        metrics = score(true_labels, preds)

        # ── step 2: compute drift scores (no true labels needed) ─
        vocab = get_vocab(pipe)
        oov = oov_share(vocab, texts)
        js = js_distance(train_pred_labels, preds, all_classes)
        oov_history.append(oov)
        js_history.append(js)

        # calibrate thresholds after enough test weeks
        if i == cfg.DRIFT_CALIBRATION_WEEKS - 1:
            thresholds = calibrate_thresholds(oov_history, js_history, sigma)

        drift_fired = False
        if thresholds is not None:
            drift_fired = is_drift(oov, js, thresholds)

        # ── step 3: true labels arrive ───────────────────────────
        f1_history.append(metrics["macro_f1"])

        # ── step 4: ask the rule ─────────────────────────────────
        state = {
            "test_week_index": i,
            "f1_history": f1_history,
            "drift_fired": drift_fired,
            "last_retrain_week_index": last_retrain_idx,
            "score_threshold": score_threshold,
        }
        should_retrain = rule_fn(state)

        # ── step 5: retrain if triggered ─────────────────────────
        train_time = 0.0
        if should_retrain:
            # sliding window: last WINDOW_SIZE weeks up to and including
            # the current test week
            all_available = train_weeks + test_weeks[: i + 1]
            window = all_available[-cfg.WINDOW_SIZE :]
            retrain_df = sample_df[sample_df["week_id"].isin(window)]
            pipe, train_time = fit_pipeline(
                retrain_df["text"].tolist(), retrain_df["label"].tolist()
            )
            # update the baseline predictions for JS distance
            train_pred_labels = predict(pipe, retrain_df["text"].tolist())
            last_retrain_idx = i

        results.append(
            {
                "week_id": wk,
                "macro_f1": metrics["macro_f1"],
                "accuracy": metrics["accuracy"],
                "oov_share": oov,
                "js_distance": js,
                "drift_fired": drift_fired,
                "retrained": should_retrain,
                "train_time_s": train_time,
            }
        )

    return pd.DataFrame(results)

