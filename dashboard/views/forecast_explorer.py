from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from dashboard.theme import TOKENS, altair_theme, inject, page_header
from src.storage.supabase_client import fetch_forecast_runs, save_forecast_run
from src.tracking.mlflow_utils import fetch_recent_runs
from src.utils.config import CONFIG

inject(accent_rails={"model_compare": "forecast", "series_chart": "forecast",
                      "experiment_history": "forecast"})

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

holdout_model = holdout["model"].iloc[0] if "model" in holdout.columns and not holdout.empty else None

prophet_holdout_rows = prophet[prophet["fold"] == "holdout"]
sarima_holdout_rows = sarima[sarima["fold"] == "holdout"]
ml_n_series = holdout[["store_nbr", "family"]].drop_duplicates().shape[0]
prophet_n_series = prophet_holdout_rows[["store_nbr", "family"]].drop_duplicates().shape[0]
sarima_n_series = sarima_holdout_rows[["store_nbr", "family"]].drop_duplicates().shape[0]

ml_holdout = (
    ml[ml["fold"] == "holdout"][["model", "mape", "wape", "mase"]]
    .assign(source="ML (global)", n_series=ml_n_series)
)
prophet_holdout = (
    prophet_holdout_rows[["mape", "wape", "mase"]]
    .mean().to_frame().T.assign(model="prophet", source="Prophet (60-series avg)", n_series=prophet_n_series)
)
sarima_holdout = (
    sarima_holdout_rows[["mape", "wape", "mase"]]
    .mean().to_frame().T.assign(model="sarima", source="SARIMA (3-series avg)", n_series=sarima_n_series)
)
comparison = pd.concat([ml_holdout, prophet_holdout, sarima_holdout], ignore_index=True)
comparison["fold"] = "holdout"
for col in ["mape", "wape", "mase"]:
    comparison[col] = comparison[col].astype(float)
comparison["n_series"] = comparison["n_series"].astype(int)
comparison = comparison.sort_values("mase").reset_index(drop=True)

best_row = comparison.iloc[0]
best_mase = float(best_row["mase"])
vs_naive_pct = (1 - best_mase) * 100

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
    chart_df["label"] = chart_df["model"]
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
                alt.value(TOKENS["chart_muted"]),
            ),
            tooltip=["source", "model", "n_series", alt.Tooltip("mase:Q", format=".3f"),
                      alt.Tooltip("mape:Q", format=".2f"), alt.Tooltip("wape:Q", format=".2f")],
        )
        .properties(height=42 * len(chart_df) + 20)
    )
    baseline_rule = (
        alt.Chart(pd.DataFrame({"x": [1.0]}))
        .mark_rule(strokeDash=[4, 3], color=TOKENS["text_faint"])
        .encode(x="x:Q")
    )
    st.altair_chart(altair_theme(bars + baseline_rule), width='stretch')
    st.caption("Dashed line marks the naive seasonal (7-day) baseline, MASE = 1.0. "
               "Models to its left beat that baseline; models to its right don't.")

    st.dataframe(
        comparison[["source", "model", "n_series", "mape", "wape", "mase"]],
        width='stretch',
        hide_index=True,
    )
    st.caption("n_series: how many store/family series each row's metrics were computed "
               "over. SARIMA only ran a full grid search on 3 representative series (CPU "
               "cost), so its row isn't directly comparable in scope to the 60-series ML "
               "and Prophet rows - lower series count means a noisier average.")
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

    with st.expander("Past logged comparisons"):
        try:
            past_runs = fetch_forecast_runs()
        except Exception as e:
            past_runs = None
            st.caption(f"Could not load past comparisons: {e}")
        if past_runs:
            st.dataframe(pd.DataFrame(past_runs), width='stretch', hide_index=True)
        elif past_runs is not None:
            st.caption("No comparisons logged yet.")

st.markdown('<div class="rc-eyebrow" style="--rc-eyebrow-color:{}">Experiment history</div>'
            .format(TOKENS["forecast"]), unsafe_allow_html=True)
st.caption("Past ML training runs tracked via MLflow on DagsHub, separate from the "
           "holdout comparison above (this pulls raw run metrics/params, not just the "
           "final holdout scores).")
with st.container(border=True, key="experiment_history"):
    try:
        recent_runs = fetch_recent_runs()
    except Exception as e:
        recent_runs = None
        st.caption(f"Could not load MLflow run history: {e}")
    if recent_runs:
        st.dataframe(pd.json_normalize(recent_runs), width='stretch', hide_index=True)
    elif recent_runs is not None:
        st.caption("No MLflow run history available. Set DAGSHUB_TOKEN and DAGSHUB_REPO "
                   "in .env to enable this (optional - everything else works without it).")

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
                x=alt.X("date:T", title=None, axis=alt.Axis(format="%b %d")),
                y=alt.Y("units:Q", title="units"),
                color=alt.Color(
                    "series:N",
                    scale=alt.Scale(
                        domain=["sales", "forecast"],
                        range=[TOKENS["text"], TOKENS["forecast"]],
                    ),
                    legend=alt.Legend(title=None, orient="top"),
                ),
                tooltip=["date:T", "series:N", alt.Tooltip("units:Q", format=".1f")],
            )
            .properties(height=320)
        )
        st.altair_chart(altair_theme(line), width='stretch')
        if holdout_model is not None:
            st.caption(f"Predictions shown are from {holdout_model}, the best of the ML "
                       f"models on holdout MASE (see comparison above for how it stacks "
                       f"up against Prophet/SARIMA).")
        else:
            st.caption("Predictions shown are from the best of the ML models on holdout "
                       "MASE (see comparison above for how it stacks up against "
                       "Prophet/SARIMA). Re-run notebook 04 to record which model this is.")
