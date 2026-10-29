"""
WildChat drift check: does the data really change from month to month?

Run:   python drift_check.py
Needs: pip install datasets pandas matplotlib scikit-learn scipy pyarrow

First run downloads the dataset (a few GB). Later runs use the cache.
If Hugging Face asks you to log in or accept terms, do that once:
    huggingface-cli login
Output: drift_check.png, drift_summary.csv, topics.txt
"""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

PER_MONTH = 3000  # English chats sampled per month for the text checks
MIN_CHATS = 2000  # skip months with fewer chats (first and last are partial)
N_TOPICS = 8
SEED = 42


def load_wildchat():
    return load_dataset("allenai/WildChat-1M", split="train")


def first_user_prompt(conversation):
    for turn in conversation:
        if turn["role"] == "user":
            return turn["content"] or ""
    return ""


def main():
    ds = load_wildchat()

    # ---- 1) cheap columns for ALL chats ----
    meta = ds.select_columns(["timestamp", "language", "model"]).to_pandas()
    meta["month"] = meta["timestamp"].dt.strftime("%Y-%m")
    counts = meta["month"].value_counts()
    months = sorted(counts[counts >= MIN_CHATS].index)
    meta = meta[meta["month"].isin(months)].copy()
    print(f"Months kept: {months[0]} to {months[-1]} ({len(months)} months)")

    # language share (top 4 languages, rest grouped as Other)
    top_langs = meta["language"].value_counts().head(4).index.tolist()
    lang = meta["language"].where(meta["language"].isin(top_langs), "Other")
    lang_share = pd.crosstab(meta["month"], lang, normalize="index") * 100
    english_share = (meta["language"] == "English").groupby(meta["month"]).mean() * 100

    # share of chats answered by a GPT-4 model
    is_gpt4 = meta["model"].str.contains("gpt-4", case=False, na=False)
    gpt4_share = is_gpt4.groupby(meta["month"]).mean() * 100

    # ---- 2) text checks on a sample of English chats ----
    eng = meta[meta["language"] == "English"]
    parts = []
    for m in months:
        g = eng[eng["month"] == m]
        parts.append(g.sample(min(len(g), PER_MONTH), random_state=SEED))
    sample = pd.concat(parts)

    rows = ds.select(sample.index.tolist()).select_columns(["conversation"])
    sample["prompt"] = [first_user_prompt(r["conversation"]) for r in rows]
    sample["words"] = sample["prompt"].str.split().str.len()
    median_len = sample.groupby("month")["words"].median()

    # topic mix: group prompts into topics with no labels needed
    vec = TfidfVectorizer(max_features=20000, stop_words="english", min_df=5)
    X = vec.fit_transform(sample["prompt"])
    km = MiniBatchKMeans(n_clusters=N_TOPICS, random_state=SEED, n_init=3)
    sample["topic"] = km.fit_predict(X)

    terms = np.array(vec.get_feature_names_out())
    with open("topics.txt", "w", encoding="utf-8") as f:
        for k in range(N_TOPICS):
            top = terms[np.argsort(km.cluster_centers_[k])[::-1][:8]]
            line = f"Topic {k}: " + ", ".join(top)
            print(line)
            f.write(line + "\n")

    topic_share = pd.crosstab(sample["month"], sample["topic"], normalize="index") * 100
    first = topic_share.iloc[0].values / 100
    topic_dist = topic_share.apply(
        lambda r: jensenshannon(first, r.values / 100, base=2), axis=1
    )  # 0 = same topic mix as the first month, 1 = totally different

    # ---- 3) charts + summary ----
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    lang_share.plot(ax=ax[0, 0], marker="o")
    ax[0, 0].set_title("Language share (% of chats)")
    gpt4_share.plot(ax=ax[0, 1], marker="o", color="tab:red")
    ax[0, 1].set_title("Chats answered by GPT-4 (%)")
    median_len.plot(ax=ax[1, 0], marker="o", color="tab:green")
    ax[1, 0].set_title("Median prompt length (words, English)")
    topic_share.plot(ax=ax[1, 1], kind="bar", stacked=True, width=0.9)
    ax[1, 1].set_title("Topic mix (English), see topics.txt")
    ax[1, 1].legend(title="Topic", fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    for a in ax.flat:
        a.set_xlabel("")
        a.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig("drift_check.png", dpi=150)

    summary = pd.DataFrame(
        {
            "english_pct": english_share,
            "gpt4_pct": gpt4_share,
            "median_words": median_len,
            "topic_distance_from_first_month": topic_dist,
        }
    )
    summary.to_csv("drift_summary.csv")

    print("\n--- How much did things move? ---")
    print(f"English share:    {english_share.min():.0f}% to {english_share.max():.0f}%")
    print(f"GPT-4 share:      {gpt4_share.min():.0f}% to {gpt4_share.max():.0f}%")
    print(f"Median prompt:    {median_len.min():.0f} to {median_len.max():.0f} words")
    print(f"Topic distance:   up to {topic_dist.max():.2f} (0 = no change, 1 = totally different)")
    print("\nSaved: drift_check.png, drift_summary.csv, topics.txt")


if __name__ == "__main__":
    main()
