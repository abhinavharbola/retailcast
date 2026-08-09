from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from dashboard.theme import TOKENS, altair_theme, inject, page_header
from src.storage.supabase_client import save_forecast_run
from src.utils.config import CONFIG

inject(accent_rails={"model_compare": "forecast", "series_chart": "forecast"})

page_header(
    eyebrow="Model benchmark - holdout window",
    title="Forecast Explorer",
    subtitle="Prophet, SARIMA, LightGBM, and XGBoost compared on the same 15-day holdout, "
             "scored with MASE (scale-free, comparable across series of very different volume).",
    accent="forecast",
)

DATA_DIR = Path(CONFIG["data"]["kaggle_outputs_dir"])
FILES = CONFIG["data"]["files"]


@st.cache_data
def load_results():
    prophet = pd.read_csv(DATA_DIR / FILES["prophet_results"])
    sarima = pd.read_csv(DATA_DIR / FILES["sarima_results"])
    ml = pd.read_csv(DATA_DIR / FILES["ml_results"])
    holdout = pd.read_parquet(DATA_DIR / FILES["final_holdout_predictions"])
    return prophet, sarima, ml, holdout


prophet, sarima, ml, holdout = load_results()

ml_holdout = ml[ml["fold"] == "holdout"][["model", "mape", "wape", "mase"]].assign(source="ML (global)")
prophet_holdout = (
    prophet[prophet["fold"] == "holdout"][["mape", "wape", "mase"]]
    .mean().to_frame().T.assign(model="prophet", source="Prophet (60-series avg)")
)
sarima_holdout = (
    sarima[sarima["fold"] == "holdout"][["mape", "wape", "mase"]]
    .mean().to_frame().T.assign(model="sarima", source="SARIMA (3-series avg)")
)
comparison = pd.concat([ml_holdout, prophet_holdout, sarima_holdout], ignore_index=True)
comparison["fold"] = "holdout"  # this view only ever compares the holdout window
for col in ["mape", "wape", "mase"]:
    comparison[col] = comparison[col].astype(float)
comparison = comparison.sort_values("mase").reset_index(drop=True)

best_row = comparison.iloc[0]
best_mase = float(best_row["mase"])
vs_naive_pct = (1 - best_mase) * 100  # MASE < 1.0 means "beats a naive seasonal forecast"

st.markdown(
    f'<div class="rc-card rc-card--forecast">'
    f'<div class="rc-card-title">Best on holdout: {str(best_row["model"]).upper()}</div>'
    f'<div class="rc-card-body">'
    f'<span class="rc-stat-value" style="font-size:1.5rem">{best_mase:.3f}</span> MASE'
    + (
        f' &nbsp;\u2022&nbsp; <span style="color:{TOKENS["good"]}">{vs_naive_pct:.1f}% lower error '
        f'than a naive seasonal (7-day) baseline</span>' if best_mase < 1
        else ' &nbsp;\u2022&nbsp; at or above the naive seasonal baseline (MASE \u2265 1.0)'
    )
    + f' &nbsp;\u2022&nbsp; {float(best_row["mape"]):.2f}% MAPE &nbsp;\u2022&nbsp; '
    f'{float(best_row["wape"]):.2f}% WAPE'
    f'</div></div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="rc-eyebrow" style="--rc-eyebrow-color:{}">Model comparison (holdout)</div>'
            .format(TOKENS["forecast"]), unsafe_allow_html=True)

with st.container(border=True, key="model_compare"):
    chart_df = comparison.copy()
    chart_df["label"] = chart_df["source"]
    chart_df["is_best"] = chart_df["model"] == best_row["model"]

    bars = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusEnd=2)
        .encode(
            y=alt.Y("label:N", sort="-x", title=None),
            x=alt.X("mase:Q", title="MASE (lower is better)"),
            color=alt.condition(
                alt.datum.is_best,
                alt.value(TOKENS["forecast"]),
                alt.value(TOKENS["neutral"]),
            ),
            tooltip=["source", "model", alt.Tooltip("mase:Q", format=".3f"),
                      alt.Tooltip("mape:Q", format=".2f"), alt.Tooltip("wape:Q", format=".2f")],
        )
        .properties(height=42 * len(chart_df) + 20)
    )
    baseline_rule = (
        alt.Chart(pd.DataFrame({"x": [1.0]}))
        .mark_rule(strokeDash=[4, 3], color=TOKENS["text_faint"])
        .encode(x="x:Q")
    )
    st.altair_chart(altair_theme(bars + baseline_rule), use_container_width=True)
    st.caption("Dashed line marks the naive seasonal (7-day) baseline, MASE = 1.0. "
               "Models to its left beat that baseline; models to its right don't.")

    st.dataframe(
        comparison[["source", "model", "mape", "wape", "mase"]],
        use_container_width=True,
        hide_index=True,
    )
    if st.button("Log this comparison to Supabase"):
        logged = 0
        error = None
        for _, row in comparison[["model", "fold", "mape", "wape", "mase"]].iterrows():
            try:
                save_forecast_run(row.to_dict())
                logged += 1
            except Exception as e:
                error = str(e)
                break
        if error:
            st.error(f"Logging failed after {logged} rows: {error}")
        else:
            st.success(f"Logged {logged} rows to Supabase.")

st.divider()
st.markdown('<div class="rc-eyebrow" style="--rc-eyebrow-color:{}">Store-family forecast vs. actual</div>'
            .format(TOKENS["forecast"]), unsafe_allow_html=True)

col1, col2 = st.columns(2)
store = col1.selectbox("Store", sorted(holdout["store_nbr"].unique()))
family = col2.selectbox("Family", sorted(holdout["family"].unique()))

series = holdout[(holdout["store_nbr"] == store) & (holdout["family"] == family)].sort_values("date")
with st.container(border=True, key="series_chart"):
    if series.empty:
        st.warning("No holdout predictions for this store/family combination.")
    else:
        melted = series.melt(id_vars="date", value_vars=["sales", "forecast"],
                              var_name="series", value_name="units")
        line = (
            alt.Chart(melted)
            .mark_line(point=True, strokeWidth=2)
            .encode(
                x=alt.X("date:T", title=None),
                y=alt.Y("units:Q", title="units"),
                color=alt.Color(
                    "series:N",
                    scale=alt.Scale(
                        domain=["sales", "forecast"],
                        range=[TOKENS["text_muted"], TOKENS["forecast"]],
                    ),
                    legend=alt.Legend(title=None, orient="top"),
                ),
                tooltip=["date:T", "series:N", alt.Tooltip("units:Q", format=".1f")],
            )
            .properties(height=320)
        )
        st.altair_chart(altair_theme(line), use_container_width=True)
        st.caption(f"Predictions shown are from {best_row['model']}, the best model on holdout MASE.")
