from datetime import datetime, timezone
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from dashboard.theme import TOKENS, altair_theme, badge, grounding_bar, grounding_pill, inject, page_header
from src.llm.grounding_check import check_grounding
from src.llm.narrative import generate_narrative
from src.storage.supabase_client import fetch_reports, save_report
from src.utils.config import CONFIG

inject(accent_rails={
    "narrative_card": "ai", "fallback_log": "ai", "model_compare_ai": "forecast",
    "anomaly_chart_0": "anomaly", "anomaly_chart_1": "anomaly",
})

page_header(
    eyebrow="GenAI results brief - grounding verified",
    title="AI-Generated Results Report",
    subtitle="Every number the model writes is checked against the same facts it was "
             "given. Nothing here is taken on faith.",
    accent="ai",
)

DATA_DIR = str(Path(CONFIG["data"]["kaggle_outputs_dir"]))
# Shared by all three "Key metrics at a glance" charts (see render_model_mase_chart and
# render_anomaly_method_chart) so their boxes come out the same height by construction.
CHART_BOX_HEIGHT = 130


def build_report_markdown(text, facts, provider, grounding_ratio, created_at=None):
    created_at = created_at or datetime.now(timezone.utc).isoformat()
    facts_lines = "\n".join(f"- **{k}**: {v}" for k, v in facts.items())
    return f"""# RetailCast AI Report

**Generated:** {created_at}
**Provider:** {provider}
**Grounding ratio:** {grounding_ratio * 100:.0f}%

---

{text}

---

## Source facts

{facts_lines}
"""


def download_filename(provider, created_at):
    safe_ts = str(created_at).replace(":", "-").replace(" ", "_")
    return f"retailcast_report_{provider}_{safe_ts}.md"


def render_model_mase_chart(facts):
    df = pd.DataFrame({
        "model": [facts["best_ml_model"], "prophet", "sarima"],
        "mase": [facts["best_ml_mase_holdout"], facts["prophet_mase_holdout"], facts["sarima_mase_holdout"]],
    })
    df["is_best"] = df["model"] == facts["best_ml_model"]
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=2)
        .encode(
            y=alt.Y("model:N", sort="-x", title=None),
            x=alt.X("mase:Q", title=None),
            color=alt.condition(alt.datum.is_best, alt.value(TOKENS["forecast"]), alt.value(TOKENS["neutral"])),
            tooltip=["model", alt.Tooltip("mase:Q", format=".3f")],
        )
        # Title lives inside the Vega-Lite spec (not an st.caption above the container)
        # so its height is identical across all three "Key metrics" boxes by construction:
        # Vega-Lite reserves the same fixed title band for any single-line title regardless
        # of its text, whereas stacking a variable number of st.caption lines above each
        # container (as this used to do) pushed some boxes' tops lower than others.
        .properties(height=CHART_BOX_HEIGHT, title="MASE by model")
    )
    return altair_theme(chart)


def render_anomaly_method_chart(precision, recall, title):
    # One chart per method (no yOffset grouping) - every bar gets its own row so axis
    # labels can't collide, regardless of how narrow the column gets.
    df = pd.DataFrame({"metric": ["precision", "recall"], "score": [precision, recall]})
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=2, size=20)
        .encode(
            y=alt.Y("metric:N", sort=["precision", "recall"], title=None),
            x=alt.X("score:Q", scale=alt.Scale(domain=[0, 1]), title=None,
                     axis=alt.Axis(values=[0, 0.2, 0.4, 0.6, 0.8, 1.0])),
            color=alt.Color(
                "metric:N",
                scale=alt.Scale(domain=["precision", "recall"], range=[TOKENS["anomaly"], TOKENS["forecast"]]),
                legend=None,
            ),
            tooltip=["metric", alt.Tooltip("score:Q", format=".2f")],
        )
        .properties(height=CHART_BOX_HEIGHT, title=title)
    )
    return altair_theme(chart)


if st.button("Generate new report", type="primary"):
    with st.spinner("Calling LLM provider..."):
        try:
            result = generate_narrative(DATA_DIR)
        except Exception as e:
            st.error(f"Report generation failed: {e}")
            st.stop()

    grounding = check_grounding(result["text"], result["facts"])
    facts = result["facts"]
    generated_at = datetime.now(timezone.utc).isoformat()

    fallback_happened = any(a["provider"] != result["provider"] for a in result["attempts"])
    with st.container(border=True, key="fallback_log"):
        with st.expander("Provider fallback log", expanded=fallback_happened):
            for a in result["attempts"]:
                status_badge = badge("ok", "good") if a["success"] else badge("failed", "bad")
                st.markdown(
                    f'{status_badge} <strong>{a["provider"]}</strong> '
                    f'<span style="color:{TOKENS["text_muted"]}">{a["error"] or "succeeded"}</span>',
                    unsafe_allow_html=True,
                )

    st.markdown('<div class="rc-eyebrow">Key metrics at a glance</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Forecasting: MASE on holdout (lower is better)")
        with st.container(border=True, key="model_compare_ai"):
            st.altair_chart(render_model_mase_chart(facts), width='stretch')
    with col2:
        st.caption("Anomaly detection: precision vs recall")
        sub1, sub2 = st.columns(2)
        with sub1:
            with st.container(border=True, key="anomaly_chart_0"):
                st.altair_chart(
                    render_anomaly_method_chart(
                        facts["control_limit_precision"], facts["control_limit_recall"], "Control limits",
                    ),
                    width='stretch',
                )
        with sub2:
            with st.container(border=True, key="anomaly_chart_1"):
                st.altair_chart(
                    render_anomaly_method_chart(
                        facts["isoforest_precision"], facts["isoforest_recall"], "Isolation Forest",
                    ),
                    width='stretch',
                )

    if "best_ml_estimated_cost_usd" in facts:
        st.markdown(
            f'<div class="rc-badge rc-badge--neutral" style="font-size:0.82rem">'
            f'Estimated cost of forecast error ({facts["best_ml_model"]}, holdout): '
            f'${facts["best_ml_estimated_cost_usd"]:,.2f}. Illustrative, based on published '
            f'margin benchmarks, not verified P&amp;L data.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:0.9rem'></div>", unsafe_allow_html=True)
    st.markdown(grounding_bar(grounding["grounded_ratio"]), unsafe_allow_html=True)

    st.markdown('<div class="rc-eyebrow" style="--rc-eyebrow-color:{}">Narrative</div>'
                .format(TOKENS["ai"]), unsafe_allow_html=True)
    with st.container(border=True, key="narrative_card"):
        st.markdown(result["text"])

    dl_col, claims_col, facts_col = st.columns(3)
    with dl_col:
        st.download_button(
            "Download this report (.md)",
            data=build_report_markdown(result["text"], facts, result["provider"], grounding["grounded_ratio"], generated_at),
            file_name=download_filename(result["provider"], generated_at),
            mime="text/markdown",
            icon=":material/download:",
            width='stretch',
        )
    with claims_col:
        with st.expander("Flagged numeric claims"):
            ungrounded = [c for c in grounding["claims"] if not c["grounded"]]
            st.write(ungrounded if ungrounded else "None. Every extracted number matched a source figure.")
    with facts_col:
        with st.expander("Source facts used"):
            st.json(facts)

    try:
        save_report(result["text"], facts, result["provider"], grounding["grounded_ratio"])
        st.caption("Saved to Supabase.")
    except Exception as e:
        st.caption(f"Not saved to Supabase: {e}")

st.divider()
st.markdown('<div class="rc-eyebrow">Past reports</div>', unsafe_allow_html=True)
try:
    past = fetch_reports()
    if past:
        for r in past:
            created_at = r.get("created_at", "unknown date")
            provider = r.get("provider", "unknown provider")
            ratio = r.get("grounding_ratio", 0)
            with st.expander(f"{created_at} \u2022 {provider} \u2022 {ratio * 100:.0f}% grounded"):
                st.markdown(grounding_pill(ratio), unsafe_allow_html=True)
                st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
                st.markdown(r["report_text"])
                st.download_button(
                    "Download this report (.md)",
                    data=build_report_markdown(
                        r["report_text"], r.get("facts", {}), provider,
                        ratio, created_at,
                    ),
                    file_name=download_filename(provider, created_at),
                    mime="text/markdown",
                    icon=":material/download:",
                    key=f"download_{r.get('id', created_at)}",
                )
    else:
        st.caption("No saved reports yet.")
except Exception as e:
    st.caption(f"Could not load past reports: {e}")