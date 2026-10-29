# Scheduled vs Drift-Triggered Retraining

**Research question:** When should a prompt-type classifier be retrained —
on a fixed schedule, or only when data drift is detected?

This project trains a TF-IDF + Logistic Regression classifier on weekly
batches from [WildChat](https://huggingface.co/datasets/sh0416/wildchat-1m-tagged)
and compares five (seven, counting sigma variants) retraining strategies.

---

## Setup

```bash
# create a virtual environment (recommended)
python -m venv .venv && source .venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

## Run

### Full experiment (5 seeds × 7 rules)

```bash
# step 1: build pool.parquet from HF (one-time, ~10 min download)
python -m src.data_loader

# step 2: run all experiments
python run_all.py
```

### Quick test (1 seed, fewer weeks)

```bash
python run_all.py --seeds 1 --max-weeks 20
```

### CI smoke test (no HF download)

```bash
python run_all.py --seeds 1 --max-weeks 18 \
  --pool data/tiny_sample.parquet --sample-dir /tmp/samples
```

### Interactive Dashboards

**1. Interactive Web Dashboard (Recommended):**
```bash
python dashboard/server.py --port 8501
# Open http://localhost:8501
```
Features:
- Dark/Light theme toggle
- Interactive performance vs. retrain cost Pareto scatter with error bars
- Dynamic multi-rule timeline with retrain markers and seed selector
- Dual unsupervised drift charts (OOV share & JS divergence) with calibration thresholds
- Searchable, paginated raw dataset explorer with CSV export

**2. Streamlit Dashboard:**
```bash
streamlit run dashboard/app.py
```

## Retraining Rules

| # | Rule | Needs labels? | Description |
|---|------|:---:|-------------|
| 1 | `never` | No | Never retrain after initial fit |
| 2 | `every_week` | No | Retrain every week |
| 3 | `every_4_weeks` | No | Retrain every 4 weeks |
| 4 | `drift_2σ/3σ/4σ` | No | Retrain when OOV share or JS distance exceeds mean + kσ threshold |
| 5 | `score_0.02/0.03/0.05` | **Yes** | Retrain when 2-week rolling macro-F1 drops > threshold below reference |

### Drift detector (rule 4)
- **OOV share:** fraction of words not in the training vocabulary.
- **JS distance:** Jensen-Shannon distance between the week's
  *predicted*-class distribution and the training period's *predicted*-class
  distribution (no true labels needed).
- **Thresholds:** calibrated from the first 6 test weeks (mean + kσ), with
  k ∈ {2, 3, 4}.
- **Cooldown:** at most one retrain every 2 weeks.

### Score trigger (rule 5)
- **Reference:** mean macro-F1 of the first 6 test weeks.
- **Check:** average of the last 2 test weeks.
- **Trigger:** fires when the check drops more than threshold ({0.02, 0.03, 0.05}) below the reference.
- **Cooldown:** 2 weeks.

## Streaming Assumption

Each week the model **predicts first**, then the true labels arrive.
The rule decides on retraining only *after* the labels are available
(except drift-triggered, which needs no labels at all).

## Multiple Seeds

The experiment uses 5 different random seeds. Each seed produces a
**different random sample of the same real timeline** — the weeks and
their order stay the same, only which ~1 000 chats are sampled changes.
The reported standard deviation therefore shows **sampling noise**, not
variation across different timelines.

## Experimental Results (5 seeds across 57 weeks)

Evaluated on 45 test weeks using a 12-week sliding window. Mean ± standard deviation across 5 random seeds (sampling noise on the same timeline):

| Rule | Needs labels? | Avg Macro F1 | Weeks Below 0.42 F1 | Avg Retrains | Avg Train Time (s) |
|------|:---:|:---:|:---:|:---:|:---:|
| `never` | No | 0.4441 ± 0.0031 | 6.2 ± 1.6 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| `every_week` | No | 0.4834 ± 0.0026 | 1.0 ± 0.0 | 45.0 ± 0.0 | 204.4 ± 8.9 |
| `every_4_weeks` | No | 0.4764 ± 0.0031 | 1.2 ± 0.4 | 11.0 ± 0.0 | 50.1 ± 3.0 |
| `drift_2σ` | No | 0.4725 ± 0.0033 | 1.2 ± 0.4 | 11.2 ± 1.1 | 52.5 ± 8.9 |
| `drift_3σ` | No | 0.4704 ± 0.0040 | 1.2 ± 0.4 | 9.2 ± 1.1 | 96.8 ± 119.1 |
| `drift_4σ` | No | 0.4664 ± 0.0025 | 1.2 ± 0.4 | 6.0 ± 2.7 | 27.9 ± 14.6 |
| `score_0.02` | **Yes** | 0.4639 ± 0.0056 | 2.0 ± 0.7 | 3.8 ± 1.5 | 197.2 ± 403.1 |
| `score_0.03` | **Yes** | 0.4597 ± 0.0054 | 2.8 ± 0.8 | 2.6 ± 0.9 | 11.7 ± 3.8 |
| `score_0.05` | **Yes** | 0.4541 ± 0.0073 | 4.0 ± 2.0 | 1.2 ± 0.8 | 5.2 ± 3.4 |

### Key Findings
1. **Retraining helps**: Continuous weekly retraining yields a **+0.039 macro F1** lift over never retraining (0.4834 vs 0.4441) and slashes weeks with poor performance ($<0.42$) from 6.2 down to 1.0.
2. **Fixed schedule (`every_4_weeks`) is the Pareto front winner**: Captures over 82% of the performance gains of weekly retraining (0.4764 vs 0.4834) with **75.6% fewer retrains** (11 vs 45) and virtually the same low rate of poor weeks (1.2 vs 1.0).
3. **Drift-triggered retraining works without ground truth**: At $2\sigma$ and $3\sigma$, drift detection triggers 9–11 retrains dynamically based solely on out-of-vocabulary rate and predicted class shift, achieving **0.470–0.473 macro F1** and limiting poor weeks to 1.2 without ever needing true labels.
4. **Tuning the score trigger**: Decreasing the score drop threshold from 0.05 to 0.02 triggers more timely retraining (3.8 vs 1.2 retrains), improving Macro F1 (0.4639 vs 0.4541) and cutting weeks below 0.42 in half (2.0 vs 4.0). However, it still lags behind label-free drift triggers (`drift_3σ`: 0.4704) because waiting for a performance dip is inherently reactive.

## Outputs

| File | Description |
|------|-------------|
| `results/results.csv` | Per-week data for every (rule, seed) pair |
| `results/summary.csv` | Per-rule aggregate: avg F1, weeks below 0.42, retrains, time, needs_labels |
| `results/f1_over_time.png` | Macro-F1 plot with retrain markers |
| `results/tradeoff.png` | Performance vs. retraining cost scatter plot with error bars |
| `results/drift_scores.png` | OOV share and JS distance with thresholds |

All runs are logged to **MLflow** via SQLite (`mlruns.db`):

```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db
```

## Limitations

1. **Noisy labels.** The `category` labels were produced by a small model
   reading the last user message. They are not human-verified.
2. **Low absolute scores.** Macro-F1 is around 0.5, so small differences
   between rules may be within noise.
3. **Single source.** All data comes from users of one free demo site
   (LMSYS ChatBot Arena). It is not representative of all LLM usage.
4. **No feature drift of the text itself** is directly measured — only
   vocabulary coverage and predicted-class distribution shifts.

## Project Structure

```
├── config.py                  # all tunables
├── run_all.py                 # entry point
├── src/
│   ├── data_loader.py         # HF → pool → per-seed samples
│   ├── model.py               # TF-IDF + LR pipeline
│   ├── drift.py               # OOV share, JS distance, thresholds
│   ├── rules.py               # 5 retraining strategies
│   ├── experiment.py          # one (rule, seed) end-to-end
│   └── viz.py                 # plots and summary tables
├── dashboard/app.py           # Streamlit dashboard
├── data/
│   ├── pool.parquet           # full cleaned pool (gitignored)
│   ├── sample_seed*.parquet   # per-seed samples (gitignored)
│   └── tiny_sample.parquet    # committed — for CI
├── results/                   # CSVs and PNGs
├── .github/workflows/ci.yml  # GitHub Actions smoke test
└── requirements.txt
```

