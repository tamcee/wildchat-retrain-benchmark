"""
Does the drift actually hurt? Train a prompt-type classifier on the first 3 months,
then test it on every later month. Also test a model retrained every month
(on the previous 3 months) to see how much retraining helps.

Run:   python drift_effect.py
Needs: pip install datasets pandas matplotlib scikit-learn pyarrow
Output: drift_effect.png, drift_effect.csv

Labels come from the community-tagged copy of WildChat. They were made by a small
model that read the LAST user message, so we use the last user message too.
"""
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

READ_PER_MONTH = 8000  # English chats read per month (before removing repeats)
KEEP_PER_MONTH = 4000  # chats kept per month for the experiment
MAX_COPIES = 5         # "deduped" setting: keep at most this many copies of the same prompt start
MIN_CHATS = 2000       # skip months with fewer chats (first and last are partial)
TRAIN_MONTHS = 3       # the frozen model learns from the first 3 months
WINDOW = 3             # the retrained model learns from the previous 3 months
SEED = 42


def load_tagged():
    return load_dataset("sh0416/wildchat-1m-tagged", split="train")


def last_user_prompt(conversation):
    for turn in reversed(conversation):
        if turn["role"] == "user":
            return turn["content"] or ""
    return ""


def read_sample(ds):
    meta = ds.select_columns(["timestamp", "language", "category"]).to_pandas()
    meta["month"] = meta["timestamp"].dt.strftime("%Y-%m")
    counts = meta["month"].value_counts()
    months = sorted(counts[counts >= MIN_CHATS].index)
    meta = meta[
        meta["month"].isin(months)
        & (meta["language"] == "English")
        & meta["category"].notna()
        & (meta["category"] != "Unknown")  # the tagging model gave up on these
    ]
    parts = [
        g.sample(min(len(g), READ_PER_MONTH), random_state=SEED)
        for _, g in meta.groupby("month")
    ]
    sample = pd.concat(parts)

    rows = ds.select(sample.index.tolist()).select_columns(["conversation"])
    sample["prompt"] = [last_user_prompt(r["conversation"]) for r in rows]
    sample = sample[sample["prompt"].str.len() > 0].copy()

    # merge small categories into "Other"
    share = sample["category"].value_counts(normalize=True)
    keep = share[share >= 0.02].index
    sample["label"] = sample["category"].where(sample["category"].isin(keep), "Other")
    print("Classes:", sorted(sample["label"].unique()))
    return sorted(sample["month"].unique()), sample


def prepare(sample, cap):
    """cap=None keeps everything (raw). cap=5 keeps at most 5 copies of the same prompt start."""
    parts = []
    for _, g in sample.groupby("month"):
        if cap is not None:
            key = g["prompt"].str[:200].str.lower().str.strip()
            g = g[g.groupby(key).cumcount() < cap]
        parts.append(g.head(KEEP_PER_MONTH))  # rows are already in random order
    return pd.concat(parts)


def fit(df):
    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    X = vec.fit_transform(df["prompt"])
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X, df["label"])
    return vec, clf


def score(vec, clf, df):
    pred = clf.predict(vec.transform(df["prompt"]))
    return (
        f1_score(df["label"], pred, average="macro", zero_division=0),
        accuracy_score(df["label"], pred),
    )


def evaluate(data, months):
    train_months = months[:TRAIN_MONTHS]
    tr = data[data["month"].isin(train_months)]
    tr_fit, tr_hold = train_test_split(tr, test_size=0.2, random_state=SEED)
    vec, clf = fit(tr_fit)
    baseline_f1, baseline_acc = score(vec, clf, tr_hold)

    out = []
    for i in range(TRAIN_MONTHS, len(months)):
        m = months[i]
        test = data[data["month"] == m]
        f1_frozen, acc_frozen = score(vec, clf, test)
        win = data[data["month"].isin(months[max(0, i - WINDOW):i])]
        v2, c2 = fit(win)
        f1_re, acc_re = score(v2, c2, test)
        out.append(
            {
                "month": m,
                "f1_frozen": f1_frozen,
                "f1_retrained": f1_re,
                "acc_frozen": acc_frozen,
                "acc_retrained": acc_re,
            }
        )
        print(f"  {m}: frozen F1 {f1_frozen:.2f} | retrained F1 {f1_re:.2f}")
    return baseline_f1, baseline_acc, pd.DataFrame(out)


def main():
    ds = load_tagged()
    months, sample = read_sample(ds)
    print(f"Months: {months[0]} to {months[-1]}. Frozen model trained on: {months[:TRAIN_MONTHS]}")

    settings = {"raw (repeats kept)": None, f"deduped (max {MAX_COPIES} copies)": MAX_COPIES}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    all_results = []
    for ax, (name, cap) in zip(axes, settings.items()):
        print(f"\n=== {name} ===")
        data = prepare(sample, cap)
        base_f1, base_acc, res = evaluate(data, months)
        res["setting"] = name
        all_results.append(res)

        ax.plot(res["month"], res["f1_frozen"], marker="o", label="Frozen (trained on first 3 months)")
        ax.plot(res["month"], res["f1_retrained"], marker="o", label=f"Retrained monthly (last {WINDOW} months)")
        ax.axhline(base_f1, color="gray", linestyle="--", label="Score on held-out early months")
        ax.set_title(f"Macro F1 by month: {name}")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=8)

        print(f"Held-out score in training months: F1 {base_f1:.2f}, accuracy {base_acc:.2f}")
        print(f"Frozen model:   first test month {res['f1_frozen'].iloc[0]:.2f}, "
              f"last {res['f1_frozen'].iloc[-1]:.2f}, worst {res['f1_frozen'].min():.2f}")
        print(f"Average F1 over test months: frozen {res['f1_frozen'].mean():.2f}, "
              f"retrained {res['f1_retrained'].mean():.2f}")

    plt.tight_layout()
    plt.savefig("drift_effect.png", dpi=150)
    pd.concat(all_results).to_csv("drift_effect.csv", index=False)
    print("\nSaved: drift_effect.png, drift_effect.csv")


if __name__ == "__main__":
    main()
