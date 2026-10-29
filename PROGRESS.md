# Project Progress: Scheduled vs Drift-Triggered Retraining

## Overview
Comparing retraining strategies for a prompt-type classifier on WildChat:
1. `never` (never retrain after week 12)
2. `every_week` (retrain weekly on sliding 12-week window)
3. `every_4_weeks` (retrain every 4 weeks)
4. `drift_triggered` (at 2σ, 3σ, 4σ thresholds; uses OOV share & JS distance of predicted-class mix; no labels; 2-week cooldown)
5. `score_triggered` (at drop thresholds 0.02, 0.03, 0.05 below first 6 test weeks reference; needs labels; 2-week cooldown)

Evaluated across 5 random seeds (sampling noise on the same timeline).

---

## Phases Status

| Phase | Description | Status | Details |
|---|---|---|---|
| **Phase 0** | Scaffolding & Setup | **COMPLETED** | Directories created, `requirements.txt` installed in `.venv`, `config.py` aligned with all approved changes, `.gitignore` configured. All imports verified. |
| **Phase 1** | Data Pipeline (`src/data_loader.py`) | **COMPLETED** | Executed and verified. Generated `data/pool.parquet` (164,100 chats, 57 weeks), 5 per-seed samples `data/sample_seed*.parquet` (~1,000 chats/week, 56,707 rows each), and `data/tiny_sample.parquet` (20 weeks × 200 chats = 4,000 rows). Exactly 10 classes kept. |
| **Phase 2** | Model Pipeline (`src/model.py`) | **COMPLETED** | TF-IDF + LogisticRegression with scoring and vocab extraction. Tested and verified on sample. |
| **Phase 3** | Drift Detector (`src/drift.py`) | **COMPLETED** | OOV share, JS distance on predicted-class distributions, calibration. Tested and verified on sample. |
| **Phase 4** | Retraining Rules (`src/rules.py`) | **COMPLETED** | All 5 strategy families (`never`, `every_week`, `every_4_weeks`, `drift_triggered` [2σ, 3σ, 4σ], `score_triggered` [0.02, 0.03, 0.05]) + cooldown logic unit-tested and verified. |
| **Phase 5** | Experiment Loop (`src/experiment.py`) | **COMPLETED** | Streaming week-by-week simulation tested end-to-end on sample. |
| **Phase 6** | Multi-seed Runner + MLflow (`run_all.py`) | **COMPLETED** | Full multi-seed run (5 seeds × 9 rule variants across 57 weeks = 45 experiment runs) logged to MLflow SQLite backend (`mlruns.db`). |
| **Phase 7** | Visualization (`src/viz.py`) | **COMPLETED** | Generated `results/summary.csv`, `results/results.csv`, `results/f1_over_time.png`, `results/drift_scores.png`, and `results/tradeoff.png`. |
| **Phase 8** | Extras (Dashboard, CI, README) | **COMPLETED** | Streamlit app (`dashboard/app.py`), GitHub Actions smoke test (`.github/workflows/ci.yml`), comprehensive `README.md` with empirical findings and trade-off analysis. |

---

## Decisions & Verification Completed
1. **Virtual Environment**: `.venv` set up in workspace with Python 3.12, all dependencies (`datasets`, `scikit-learn`, `mlflow`, `streamlit`, `plotly`, `pyarrow`, etc.) installed and verified.
2. **Dataset Inspection**: Checked `sh0416/wildchat-1m-tagged`. Found 459,299 English non-Unknown rows across 57 weeks (`2023-W14` through `2024-W18`), all having >= 500 chats. Classes >= 2% yield exactly 9 classes + "Other" = 10 classes, matching `drift_effect.py`.
3. **Data Pipeline Enhancements**: Ensured sorting by timestamp across `pool.parquet` and `tiny_sample.parquet`, and confirmed dedupe & capping match specifications.
4. **MLflow Tracking Backend**: MLflow 3.x deprecates file store; configured local SQLite database (`sqlite:///mlruns.db`), added to `.gitignore`.
5. **Matplotlib & Formatting**: Fixed `MPLCONFIGDIR` to avoid sandbox cache issues; refined `viz.py` to format standard deviations cleanly.
6. **All Experiments Executed**: Successfully completed 45 end-to-end runs (5 seeds × 9 rule variants across 57 weeks). Results confirmed that `every_4_weeks` and `drift_2σ/3σ` deliver strong gains while reducing retrains by 75–80%.
7. **Score Trigger Variants & Tradeoff Plot**: Added score drop thresholds at 0.02, 0.03, and 0.05. Replaced "worst week F1" with "weeks below 0.42 F1" in `summary.csv`. Generated `results/tradeoff.png` showing Macro F1 vs. Retrain count with standard deviation error bars, integrated into `dashboard/app.py` and `README.md`.
8. **Interactive Web Dashboard Built**: Designed a high-performance web dashboard in `dashboard/web/index.html` (Inter font, Plotly charts, dark/light mode toggle, KPI cards, interactive Pareto scatter, timeline with retrain event glyphs, drift signal monitors, strategy deep dive, and paginated data explorer with CSV export). Served via zero-dependency `dashboard/server.py` on port 8501. Also upgraded `dashboard/app.py` with custom dark-zinc CSS styling.
