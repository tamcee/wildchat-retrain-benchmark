"""
drift.py — drift detection: OOV share + Jensen-Shannon distance.

OOV share:  fraction of words in the new text that are not in the
            training TF-IDF vocabulary.
JS distance: Jensen-Shannon divergence between the *predicted*-class
             distribution of the current week and the *predicted*-class
             distribution of the training period.

Thresholds are calibrated from the first N test weeks (mean + k*sigma).
"""
import re
import numpy as np
from scipy.spatial.distance import jensenshannon

import config as cfg


# ── OOV (out-of-vocabulary) share ────────────────────────────────────

_WORD_RE = re.compile(r"\b\w+\b")


def _tokenize(text):
    """Lowercase word tokenization matching TfidfVectorizer defaults."""
    return _WORD_RE.findall(text.lower())


def oov_share(vocab_set, texts):
    """
    Fraction of word tokens in `texts` that are absent from `vocab_set`.
    Returns a float between 0 and 1.
    """
    total = 0
    missing = 0
    for t in texts:
        words = _tokenize(t)
        total += len(words)
        missing += sum(1 for w in words if w not in vocab_set)
    return missing / max(total, 1)


# ── Jensen-Shannon distance on predicted-class distributions ─────────

def _class_distribution(labels, all_classes):
    """
    Turn a list/array of predicted labels into a probability vector
    aligned to `all_classes`.
    """
    counts = {}
    for lbl in labels:
        counts[lbl] = counts.get(lbl, 0) + 1
    n = max(len(labels), 1)
    return np.array([counts.get(c, 0) / n for c in all_classes])


def js_distance(train_pred_labels, week_pred_labels, all_classes):
    """
    Jensen-Shannon distance between predicted-class distributions.
    Both sides use *predicted* labels (no true labels needed).
    """
    p = _class_distribution(train_pred_labels, all_classes)
    q = _class_distribution(week_pred_labels, all_classes)
    return float(jensenshannon(p, q))


# ── Threshold calibration ───────────────────────────────────────────

def calibrate_thresholds(oov_history, js_history, sigma=3.0):
    """
    Set thresholds from the first DRIFT_CALIBRATION_WEEKS test weeks.
    threshold = mean + sigma * std

    Returns dict with keys: oov_threshold, js_threshold.
    """
    n = cfg.DRIFT_CALIBRATION_WEEKS
    oov_arr = np.array(oov_history[:n])
    js_arr = np.array(js_history[:n])
    return {
        "oov_threshold": float(oov_arr.mean() + sigma * oov_arr.std()),
        "js_threshold": float(js_arr.mean() + sigma * js_arr.std()),
    }


def is_drift(oov_val, js_val, thresholds):
    """True if either score exceeds its threshold."""
    return (oov_val > thresholds["oov_threshold"] or
            js_val > thresholds["js_threshold"])
