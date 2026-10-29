"""
model.py — TF-IDF + LogisticRegression pipeline.

Reuses the exact same settings as drift_effect.py:
  TfidfVectorizer(max_features=50000, ngram_range=(1,2), min_df=3, sublinear_tf=True)
  LogisticRegression(class_weight="balanced", max_iter=1000)
"""
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

import config as cfg


def build_pipeline():
    """Create an unfitted TF-IDF + LR pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=cfg.TFIDF_MAX_FEATURES,
            ngram_range=cfg.TFIDF_NGRAM,
            min_df=cfg.TFIDF_MIN_DF,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=cfg.LR_MAX_ITER,
        )),
    ])


def fit_pipeline(texts, labels):
    """
    Build and fit a fresh pipeline. Returns (pipeline, training_time_seconds).
    """
    pipe = build_pipeline()
    t0 = time.time()
    pipe.fit(texts, labels)
    elapsed = time.time() - t0
    return pipe, elapsed


def predict(pipe, texts):
    """Return predicted labels."""
    return pipe.predict(texts)


def predict_proba(pipe, texts):
    """Return predicted class probabilities (n_samples × n_classes)."""
    return pipe.predict_proba(texts)


def score(y_true, y_pred):
    """Compute macro F1 and accuracy. Returns a dict."""
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def get_vocab(pipe):
    """Return the set of words in the fitted TF-IDF vocabulary."""
    return set(pipe.named_steps["tfidf"].vocabulary_.keys())


def get_classes(pipe):
    """Return the array of class labels the classifier knows."""
    return pipe.named_steps["clf"].classes_
