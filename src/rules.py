"""
rules.py — five retraining strategies.

Each rule function receives the experiment state and returns True if the
model should be retrained this week.

Rules:
  1. never          — never retrain
  2. every_week     — retrain every week
  3. every_4_weeks  — retrain every 4 weeks
  4. drift_triggered — retrain when OOV or JS drift exceeds threshold (no labels)
  5. score_triggered — retrain when recent macro-F1 drops (needs labels)

Rules 4 and 5 both enforce a 2-week cooldown between retrains.
"""
import numpy as np
import config as cfg


def never(state):
    """Rule 1: never retrain after the initial fit."""
    return False


def every_week(state):
    """Rule 2: retrain every single week."""
    return True


def every_4_weeks(state):
    """Rule 3: retrain every 4 weeks."""
    weeks_since_start = state["test_week_index"]
    return weeks_since_start > 0 and weeks_since_start % 4 == 0


def drift_triggered(state):
    """
    Rule 4: retrain when drift detector fires.
    Uses NO labels — only OOV share and JS distance of predicted classes.
    Enforces a cooldown of DRIFT_COOLDOWN_WEEKS between retrains.
    """
    # need at least DRIFT_CALIBRATION_WEEKS to have thresholds
    if state["test_week_index"] < cfg.DRIFT_CALIBRATION_WEEKS:
        return False
    # cooldown check
    if _in_cooldown(state):
        return False
    return state["drift_fired"]


def score_triggered(state):
    """
    Rule 5: retrain when the average macro-F1 of the last 2 test weeks
    falls more than SCORE_DROP_THRESHOLD below the reference F1.

    Reference = mean macro-F1 of the first SCORE_REFERENCE_WEEKS test weeks.
    Enforces a 2-week cooldown.
    """
    idx = state["test_week_index"]
    f1_history = state["f1_history"]  # list of macro-F1 values so far

    # need enough weeks to compute reference and recent average
    if idx < cfg.SCORE_REFERENCE_WEEKS:
        return False
    if len(f1_history) < cfg.SCORE_CHECK_WINDOW:
        return False
    # cooldown check
    if _in_cooldown(state):
        return False

    threshold = state.get("score_threshold", cfg.SCORE_DROP_THRESHOLD)
    reference = np.mean(f1_history[:cfg.SCORE_REFERENCE_WEEKS])
    recent = np.mean(f1_history[-cfg.SCORE_CHECK_WINDOW:])
    return recent < (reference - threshold)


# ── helper ───────────────────────────────────────────────────────────

def _in_cooldown(state):
    """True if the last retrain was fewer than DRIFT_COOLDOWN_WEEKS ago."""
    last = state.get("last_retrain_week_index")
    if last is None:
        return False
    return (state["test_week_index"] - last) < cfg.DRIFT_COOLDOWN_WEEKS


# ── registry ─────────────────────────────────────────────────────────

# rule name → (function, needs_labels)
RULES = {
    "never":            (never,           False),
    "every_week":       (every_week,      False),
    "every_4_weeks":    (every_4_weeks,   False),
    "drift_triggered":  (drift_triggered, False),
    "score_triggered":  (score_triggered, True),
}
