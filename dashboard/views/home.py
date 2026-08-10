import streamlit as st

from dashboard.theme import inject, page_header, stat_row

NAV_CARDS = [
    ("views/overview.py", "Overview", "neutral",
     "Dataset scope, demand pattern classification, stationarity tests."),
    ("views/forecast_explorer.py", "Forecast Explorer", "forecast",
     "Model comparison and holdout forecast vs. actual, by store and family."),
    ("views/anomaly_view.py", "Anomaly View", "anomaly",
     "Control limits vs. Isolation Forest, synthetic-injection evaluation."),
    ("views/ai_report.py", "AI Report", "ai",
     "Grounded GenAI narrative with numeric claim verification."),
]

inject(accent_rails={f"nav_{i}": accent for i, (_, _, accent, _) in enumerate(NAV_CARDS)})

page_header(
    eyebrow="Retail forecasting & anomaly intelligence",
    title="RetailCast",
    subtitle=(
        "Demand forecasting and anomaly detection benchmarked across Prophet, SARIMA, "
        "LightGBM, and XGBoost on 60 store-family series from the Favorita Store Sales "
        "dataset - paired with a GenAI results narrative that verifies its own numbers "
        "against the underlying data before you read them."
    ),
)

stat_row([
    ("60", "series covered"),
    ("4", "models benchmarked"),
    ("2", "anomaly detection methods"),
    ("15d", "walk-forward holdout"),
])

st.markdown("<div style='height: 1.6rem'></div>", unsafe_allow_html=True)
st.markdown('<div class="rc-eyebrow">Explore</div>', unsafe_allow_html=True)

# Real st.container(border=True) blocks, not raw HTML - markdown HTML can't hold a live
# Streamlit widget, and st.page_link needs to render genuinely inside the box.
cols = st.columns(4)
for i, (col, (path, title, accent, caption)) in enumerate(zip(cols, NAV_CARDS)):
    with col:
        with st.container(border=True, key=f"nav_{i}"):
            st.markdown(
                f'<div class="rc-card-title">{title}</div>'
                f'<div class="rc-card-body">{caption}</div>',
                unsafe_allow_html=True,
            )
            st.page_link(path, label="Open", icon=":material/arrow_forward:")

st.divider()
st.caption(
    "10 stores \u00d7 6 product families \u2022 15-day walk-forward-validated forecast horizon \u2022 "
    "expanding-window CV, 4 folds + holdout"
)
