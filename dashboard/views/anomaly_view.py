from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from dashboard.theme import TOKENS, altair_theme, inject, page_header
from src.storage.supabase_client import save_anomaly_flags
from src.utils.config import CONFIG

inject(accent_rails={"eval_chart_0": "anomaly", "eval_chart_1": "anomaly", "flagged_table": "anomaly"})

page_header(
    eyebrow="Anomaly detection - synthetic-injection eval",
    title="Anomaly Detection",
    subtitle="Two detection methods scored by injecting synthetic spikes/drops into "
             "holdout residuals and checking what each method catches.",
    accent="anomaly",
)

DATA_DIR = Path(CONFIG["data"]["kaggle_outputs_dir"])
FILES = CONFIG["data"]["files"]


@st.cache_data
def load_anomaly_data():
    results = pd.read_parquet(DATA_DIR / FILES["anomaly_results"])
    eval_metrics = pd.read_csv(DATA_DIR / FILES["anomaly_eval_metrics"], index_col=0)
    return results, eval_metrics


results, eval_metrics = load_anomaly_data()

METHOD_LABELS = {
    "control_limit_flag_injected": ("Control limits", "Per-series threshold at k\u00d7std of clean residuals."),
    "isoforest_flag_injected": ("Isolation Forest", "Contamination fixed at 5%, so recall is structurally "
                                                      "capped by that rate regardless of the true anomaly count."),
}

st.markdown('<div class="rc-eyebrow" style="--rc-eyebrow-color:{}">Synthetic-injection evaluation</div>'
            .format(TOKENS["anomaly"]), unsafe_allow_html=True)

method_cols = st.columns(len(eval_metrics))
for i, (col, (method, row)) in enumerate(zip(method_cols, eval_metrics.iterrows())):
    label, note = METHOD_LABELS.get(method, (method, ""))
    with col:
        st.markdown(
            f'<div class="rc-card rc-card--anomaly rc-compare-card">'
            f'<div class="rc-card-title">{label}</div>'
            f'<div style="display:flex; gap:1.4rem; margin: 0.5rem 0 0.3rem 0;">'
            f'<div><div class="rc-stat-value" style="font-size:1.4rem">{row["precision"]:.2f}</div>'
            f'<div class="rc-stat-label">precision</div></div>'
            f'<div><div class="rc-stat-value" style="font-size:1.4rem">{row["recall"]:.2f}</div>'
            f'<div class="rc-stat-label">recall</div></div>'
            f'<div><div class="rc-stat-value" style="font-size:1.4rem">{row["f1"]:.2f}</div>'
            f'<div class="rc-stat-label">f1</div></div>'
            f'</div><div class="rc-card-body">{note}</div></div>',
            unsafe_allow_html=True,
        )
        # One chart per method (rather than one grouped chart for both) - keeps every
        # bar on its own row with no offset/banding, so axis labels can't collide
        # regardless of how narrow the column gets.
        with st.container(border=True, key=f"eval_chart_{i}"):
            metric_df = pd.DataFrame({
                "metric": ["precision", "recall", "f1"],
                "score": [row["precision"], row["recall"], row["f1"]],
            })
            chart = (
                alt.Chart(metric_df)
                .mark_bar(cornerRadiusEnd=2, size=20)
                .encode(
                    y=alt.Y("metric:N", sort=["precision", "recall", "f1"], title=None),
                    x=alt.X("score:Q", scale=alt.Scale(domain=[0, 1]), title=None,
                            axis=alt.Axis(values=[0, 0.2, 0.4, 0.6, 0.8, 1.0])),
                    color=alt.Color(
                        "metric:N",
                        scale=alt.Scale(
                            domain=["precision", "recall", "f1"],
                            range=[TOKENS["anomaly"], TOKENS["forecast"], TOKENS["text_muted"]],
                        ),
                        legend=None,
                    ),
                    tooltip=["metric", alt.Tooltip("score:Q", format=".2f")],
                )
                .properties(height=130)
            )
            st.altair_chart(altair_theme(chart), use_container_width=True)

st.divider()
st.markdown('<div class="rc-eyebrow" style="--rc-eyebrow-color:{}">Flagged anomalies on real holdout data</div>'
            .format(TOKENS["anomaly"]), unsafe_allow_html=True)

both_flagged = int(((results["control_limit_flag"] == 1) & (results["isoforest_flag"] == 1)).sum())
c1, c2, c3 = st.columns(3)
c1.metric("Control-limit flags", int(results["control_limit_flag"].sum()))
c2.metric("IsolationForest flags", int(results["isoforest_flag"].sum()))
c3.metric("Flagged by both", both_flagged)
st.caption("Points flagged by both methods are the higher-confidence anomalies. "
           "Highlighted below when viewing 'either'.")

method = st.radio("Filter by method", ["control_limit_flag", "isoforest_flag", "either"], horizontal=True)
if method == "either":
    flagged = results[(results["control_limit_flag"] == 1) | (results["isoforest_flag"] == 1)]
else:
    flagged = results[results[method] == 1]

with st.container(border=True, key="flagged_table"):
    display_cols = ["date", "store_nbr", "family", "sales", "forecast", "residual",
                     "control_limit_flag", "isoforest_flag"]
    flagged_display = flagged[display_cols].sort_values("date")

    if method == "either":
        def _co_detect_style(row):
            agree = row["control_limit_flag"] == 1 and row["isoforest_flag"] == 1
            style = f"background-color: {TOKENS['anomaly']}22" if agree else ""
            return [style] * len(row)
        st.dataframe(flagged_display.style.apply(_co_detect_style, axis=1), use_container_width=True)
    else:
        st.dataframe(flagged_display, use_container_width=True)

    if st.button("Log flagged anomalies to Supabase"):
        to_log = flagged_display.copy()
        to_log["date"] = to_log["date"].dt.strftime("%Y-%m-%d")
        try:
            save_anomaly_flags(to_log)
            st.success(f"Logged {len(to_log)} rows to Supabase.")
        except Exception as e:
            st.error(f"Logging failed: {e}")