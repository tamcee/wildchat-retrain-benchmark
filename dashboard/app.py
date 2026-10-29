"""
Streamlit dashboard — continual retraining benchmark.

Run:  streamlit run dashboard/app.py
For the standalone Web Dashboard:  python dashboard/server.py
"""
import os
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

st.set_page_config(
    page_title="Retrain Benchmark — Scheduled vs Drift-Triggered",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Design System CSS Injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Sleek Dark Zinc Cards */
    .metric-card {
        background: #18181b;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #a1a1aa;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #fafafa;
        margin-top: 4px;
        letter-spacing: -0.02em;
    }
    .metric-sub {
        font-size: 0.75rem;
        color: #71717a;
        margin-top: 6px;
    }
    .badge-pareto {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.25);
        padding: 2px 8px;
        border-radius: 9999px;
        font-size: 0.7rem;
        font-weight: 600;
    }
    .badge-blue {
        background: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.25);
        padding: 2px 8px;
        border-radius: 9999px;
        font-size: 0.7rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div style="display: flex; align-items: center; justify-content: space-between; border-b: 1px solid #27272a; padding-bottom: 1rem; margin-bottom: 1.5rem;">
    <div>
        <h1 style="font-size: 1.75rem; font-weight: 700; letter-spacing: -0.03em; margin: 0; color: #fafafa;">
            ⚡ WildChat Prompt-Type Retraining Benchmark
        </h1>
        <p style="font-size: 0.85rem; color: #a1a1aa; margin-top: 0.25rem;">
            Empirical evaluation of 9 scheduled, drift-triggered, and score-triggered retraining rules over 45 test weeks.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    results = pd.read_csv("results/results.csv")
    summary = pd.read_csv("results/summary.csv")
    return results, summary


try:
    results, summary = load_data()
except FileNotFoundError:
    st.error("No results found. Run `python run_all.py` first.")
    st.stop()

# ── Sidebar Filters ──────────────────────────────────────────────
st.sidebar.markdown("### ⚙️ Benchmark Filters")
all_rules = sorted(results["rule"].unique())
selected_rules = st.sidebar.multiselect("Strategies", all_rules, default=all_rules)

all_seeds = sorted(results["seed"].unique())
selected_seeds = st.sidebar.multiselect("Seeds", all_seeds, default=all_seeds)

filtered = results[
    results["rule"].isin(selected_rules) & results["seed"].isin(selected_seeds)
]

# ── KPI Cards ────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("""
    <div class="metric-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span class="metric-label">Pareto Champion</span>
            <span class="badge-pareto">Optimal ROI</span>
        </div>
        <div class="metric-value">every_4_weeks</div>
        <div class="metric-sub"><strong style="color: #34d399;">0.4764 F1</strong> · 11 Retrains (82% gains)</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="metric-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span class="metric-label">Performance Ceiling</span>
            <span class="badge-blue">Max F1</span>
        </div>
        <div class="metric-value">every_week</div>
        <div class="metric-sub"><strong style="color: #818cf8;">0.4834 F1</strong> · 45 Retrains (204.4s)</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="metric-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span class="metric-label">Best Label-Free</span>
            <span class="badge-pareto">No Labels</span>
        </div>
        <div class="metric-value">drift_2σ</div>
        <div class="metric-sub"><strong style="color: #38bdf8;">0.4725 F1</strong> · 11.2 Retrains (Zero True Labels)</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown("""
    <div class="metric-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span class="metric-label">Baseline Decay</span>
            <span style="background: rgba(244,63,94,0.15); color: #fb7185; border: 1px solid rgba(244,63,94,0.25); padding: 2px 8px; border-radius: 9999px; font-size: 0.7rem; font-weight: 600;">Severe</span>
        </div>
        <div class="metric-value">never</div>
        <div class="metric-sub"><strong style="color: #fb7185;">0.4441 F1</strong> · 6.2 Degraded Weeks (&lt;0.42)</div>
    </div>
    """, unsafe_allow_html=True)

# ── Summary Table ────────────────────────────────────────────────
st.markdown("### 📋 Strategy Performance Summary")
st.dataframe(
    summary[summary["rule"].isin(selected_rules)],
    use_container_width=True,
    hide_index=True,
)

# ── Tradeoff Scatter ─────────────────────────────────────────────
st.markdown("### ⚖️ Performance vs. Retraining Cost Trade-off")
if os.path.exists("results/tradeoff.png"):
    st.image("results/tradeoff.png", use_container_width=True)

# ── Macro F1 Over Time ──────────────────────────────────────────
st.markdown("### 📈 Macro F1 Trajectory Over 45 Test Weeks")

agg = (
    filtered.groupby(["rule", "week_id"])
    .agg(f1_mean=("macro_f1", "mean"), f1_std=("macro_f1", "std"),
         retrain_frac=("retrained", "mean"))
    .reset_index()
)

fig = px.line(
    agg, x="week_id", y="f1_mean", color="rule",
    labels={"f1_mean": "Macro F1 (mean across seeds)", "week_id": "Week"},
    template="plotly_dark",
    color_discrete_map={
        "every_week": "#6366f1",
        "every_4_weeks": "#10b981",
        "drift_2σ": "#06b6d4",
        "drift_3σ": "#0ea5e9",
        "drift_4σ": "#38bdf8",
        "score_0.02": "#f59e0b",
        "score_0.03": "#f97316",
        "score_0.05": "#ef4444",
        "never": "#71717a"
    }
)
for rule in selected_rules:
    rule_agg = agg[(agg["rule"] == rule) & (agg["retrain_frac"] > 0.5)]
    if len(rule_agg) > 0:
        fig.add_trace(go.Scatter(
            x=rule_agg["week_id"], y=rule_agg["f1_mean"],
            mode="markers", marker=dict(symbol="diamond", size=7, line=dict(width=1, color="white")),
            name=f"{rule} (retrain)", showlegend=False,
        ))

fig.add_hline(y=0.42, line_dash="dash", line_color="rgba(239, 68, 68, 0.7)", annotation_text="0.42 Low F1 Alert")
fig.update_layout(
    xaxis_tickangle=-45,
    height=480,
    margin=dict(l=20, r=20, t=30, b=20),
    plot_bgcolor="#121215",
    paper_bgcolor="#121215",
    font=dict(family="Inter", color="#a1a1aa")
)
st.plotly_chart(fig, use_container_width=True)

# ── Drift Scores ────────────────────────────────────────────────
st.markdown("### 🔍 Unsupervised Drift Signals (Baseline Timeline)")
col1, col2 = st.columns(2)

never_data = filtered[filtered["rule"] == "never"] if "never" in selected_rules else filtered
if len(never_data) > 0:
    drift_agg = (
        never_data.groupby("week_id")
        .agg(oov_mean=("oov_share", "mean"), js_mean=("js_distance", "mean"))
        .reset_index()
    )
    with col1:
        fig_oov = px.line(
            drift_agg, x="week_id", y="oov_mean",
            title="Out-of-Vocabulary (OOV) Share",
            labels={"oov_mean": "OOV Share"},
            template="plotly_dark",
            color_discrete_sequence=["#06b6d4"]
        )
        fig_oov.update_layout(
            xaxis_tickangle=-45, height=350,
            plot_bgcolor="#121215", paper_bgcolor="#121215",
            font=dict(family="Inter", color="#a1a1aa")
        )
        st.plotly_chart(fig_oov, use_container_width=True)

    with col2:
        fig_js = px.line(
            drift_agg, x="week_id", y="js_mean",
            title="Jensen-Shannon (JS) Distance",
            labels={"js_mean": "JS Distance"},
            template="plotly_dark",
            color_discrete_sequence=["#10b981"]
        )
        fig_js.update_layout(
            xaxis_tickangle=-45, height=350,
            plot_bgcolor="#121215", paper_bgcolor="#121215",
            font=dict(family="Inter", color="#a1a1aa")
        )
        st.plotly_chart(fig_js, use_container_width=True)

# ── Raw Data ────────────────────────────────────────────────────
with st.expander("🔎 Raw Records Explorer"):
    st.dataframe(filtered, use_container_width=True, hide_index=True)
