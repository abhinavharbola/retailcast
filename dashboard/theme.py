"""
RetailCast design system.

One place for color tokens, typography, and small render helpers so every page reads as
one product instead of five separate Streamlit defaults. The core idea: color is a legend,
not decoration. Every card gets a 3px left edge in one of four colors, and that color means
the same thing everywhere in the app:

    gray   -> dataset / configuration (ground truth about what was run)
    teal   -> forecast results (model output, backtested)
    amber  -> anomaly detection results (flags, control limits)
    violet -> AI-generated narrative (the one thing in the app that isn't a direct
              computation - it's the LLM's interpretation, verified but still generated)

Import `inject()` once per page (cheap, idempotent) and use the helpers below instead of
raw st.markdown/st.metric calls for anything that should carry that color coding.
"""

import streamlit as st

TOKENS = {
    "bg": "#0A0E13",
    "surface": "#12171F",
    "surface_hover": "#171E29",
    "border": "#232B36",
    "text": "#E7EBF0",
    "text_muted": "#8D97A6",
    "text_faint": "#56606E",
    "forecast": "#2FB8A6",
    "anomaly": "#E3A008",
    "ai": "#7C6FF0",
    "good": "#35C87A",
    "bad": "#F1554C",
    "neutral": "#4C5A6B",
}

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

/* ---- page chrome ---- */
.stApp {{
    background: {TOKENS["bg"]};
}}
section[data-testid="stSidebar"] {{
    background: {TOKENS["surface"]};
    border-right: 1px solid {TOKENS["border"]};
}}
hr {{
    border: none;
    border-top: 1px solid {TOKENS["border"]};
    margin: 1.4rem 0;
}}

/* ---- headers ---- */
.rc-eyebrow {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--rc-eyebrow-color, {TOKENS["text_muted"]});
    margin-bottom: 0.35rem;
}}
.rc-h1 {{
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 2.1rem;
    line-height: 1.15;
    color: {TOKENS["text"]};
    margin: 0 0 0.4rem 0;
}}
.rc-sub {{
    font-family: 'Inter', sans-serif;
    font-size: 0.98rem;
    color: {TOKENS["text_muted"]};
    max-width: 46rem;
    margin: 0 0 1.1rem 0;
}}

/* ---- cards (static / text content only - can't hold live widgets) ---- */
.rc-card {{
    background: {TOKENS["surface"]};
    border: 1px solid {TOKENS["border"]};
    border-left: 3px solid {TOKENS["neutral"]};
    border-radius: 4px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.9rem;
}}
.rc-card--forecast {{ border-left-color: {TOKENS["forecast"]}; }}
.rc-card--anomaly  {{ border-left-color: {TOKENS["anomaly"]}; }}
.rc-card--ai       {{ border-left-color: {TOKENS["ai"]}; }}
.rc-card--neutral  {{ border-left-color: {TOKENS["neutral"]}; }}

.rc-card-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 1.02rem;
    color: {TOKENS["text"]};
    margin-bottom: 0.2rem;
}}
.rc-card-body {{
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem;
    color: {TOKENS["text_muted"]};
    line-height: 1.5;
}}

/* ---- nav cards on Home (clickable via st.page_link, styled through the wrapper) ---- */
.rc-nav-card {{
    background: {TOKENS["surface"]};
    border: 1px solid {TOKENS["border"]};
    border-left: 3px solid var(--rc-nav-accent, {TOKENS["neutral"]});
    border-radius: 4px;
    padding: 1rem 1.2rem;
    height: 100%;
}}
.rc-nav-card .rc-card-title {{ display: flex; align-items: center; gap: 0.5rem; }}

/* ---- stat blocks (custom, mono numerals) ---- */
.rc-stat-value {{
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    font-size: 1.9rem;
    color: {TOKENS["text"]};
    line-height: 1.1;
}}
.rc-stat-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {TOKENS["text_faint"]};
    margin-top: 0.25rem;
}}

/* ---- badges / chips ---- */
.rc-badge {{
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74rem;
    font-weight: 500;
    padding: 0.14rem 0.55rem;
    border-radius: 3px;
    margin: 0.1rem 0.28rem 0.1rem 0;
    border: 1px solid transparent;
    white-space: nowrap;
}}
.rc-badge--neutral  {{ background: rgba(76,90,107,0.18);  color: {TOKENS["text_muted"]}; border-color: {TOKENS["border"]}; }}
.rc-badge--forecast {{ background: rgba(47,184,166,0.12); color: {TOKENS["forecast"]}; border-color: rgba(47,184,166,0.35); }}
.rc-badge--anomaly  {{ background: rgba(227,160,8,0.12);  color: {TOKENS["anomaly"]}; border-color: rgba(227,160,8,0.35); }}
.rc-badge--ai       {{ background: rgba(124,111,240,0.12);color: {TOKENS["ai"]}; border-color: rgba(124,111,240,0.35); }}
.rc-badge--good     {{ background: rgba(53,200,122,0.12); color: {TOKENS["good"]}; border-color: rgba(53,200,122,0.35); }}
.rc-badge--bad      {{ background: rgba(241,85,76,0.12);  color: {TOKENS["bad"]}; border-color: rgba(241,85,76,0.35); }}

/* ---- status pill (grounding ratio, etc: bigger, standalone) ---- */
.rc-pill {{
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    font-size: 0.95rem;
    padding: 0.4rem 0.85rem;
    border-radius: 4px;
    border: 1px solid;
}}
.rc-pill--good {{ background: rgba(53,200,122,0.10); color: {TOKENS["good"]}; border-color: rgba(53,200,122,0.4); }}
.rc-pill--warn {{ background: rgba(227,160,8,0.10);  color: {TOKENS["anomaly"]}; border-color: rgba(227,160,8,0.4); }}
.rc-pill--bad  {{ background: rgba(241,85,76,0.10);  color: {TOKENS["bad"]}; border-color: rgba(241,85,76,0.4); }}
.rc-pill-dot {{ width: 7px; height: 7px; border-radius: 50%; background: currentColor; }}

/* ---- native Streamlit widgets, restyled to match ---- */
[data-testid="stMetricValue"] {{
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
}}
[data-testid="stMetricLabel"] {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {TOKENS["text_muted"]} !important;
}}
[data-testid="stDataFrame"] {{
    border: 1px solid {TOKENS["border"]};
    border-radius: 4px;
}}

/* container-key targeted accent rails (bordered st.container blocks that must hold
   live widgets/dataframes/charts, so can't be plain markdown HTML) */
{{accent_rail_rules}}
</style>
"""


def inject(accent_rails: dict[str, str] | None = None) -> None:
    """Injects the shared stylesheet. Call once near the top of every page.

    accent_rails: optional {container_key: token_name} map. For any st.container(border=True,
    key=...) on the page that should carry a colored left edge, pass its key and which
    token color it should use (e.g. {"model_compare": "forecast"}). Uses Streamlit's
    `st-key-<key>` class, which is the documented hook for targeting a specific container.
    """
    rules = []
    if accent_rails:
        for key, token in accent_rails.items():
            color = TOKENS.get(token, TOKENS["neutral"])
            rules.append(
                f'.st-key-{key} {{ border-left: 3px solid {color} !important; '
                f'border-radius: 4px !important; }}'
            )
    css = _CSS.replace("{accent_rail_rules}", "\n".join(rules))
    st.markdown(css, unsafe_allow_html=True)


def page_header(eyebrow: str, title: str, subtitle: str = "", accent: str = "neutral") -> None:
    color = TOKENS.get(accent, TOKENS["neutral"])
    sub_html = f'<p class="rc-sub">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f'<div class="rc-eyebrow" style="--rc-eyebrow-color:{color}">{eyebrow}</div>'
        f'<div class="rc-h1">{title}</div>'
        f'{sub_html}',
        unsafe_allow_html=True,
    )


def card(title: str, body: str, accent: str = "neutral") -> None:
    """Static text-only card. Cannot contain live Streamlit widgets - use a
    st.container(border=True, key=...) + inject(accent_rails=...) for that instead."""
    st.markdown(
        f'<div class="rc-card rc-card--{accent}">'
        f'<div class="rc-card-title">{title}</div>'
        f'<div class="rc-card-body">{body}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def badge(text: str, kind: str = "neutral") -> str:
    """Returns an inline badge span - compose these into a row and pass to st.markdown."""
    return f'<span class="rc-badge rc-badge--{kind}">{text}</span>'


def badge_row(items: list[str], kind: str = "neutral") -> None:
    st.markdown("".join(badge(str(i), kind) for i in items), unsafe_allow_html=True)


def stat(value: str, label: str) -> str:
    """Returns HTML for one custom stat block (mono value + eyebrow label)."""
    return (
        f'<div><div class="rc-stat-value">{value}</div>'
        f'<div class="rc-stat-label">{label}</div></div>'
    )


def stat_row(stats: list[tuple[str, str]]) -> None:
    """Renders several stat() blocks evenly spaced in one row."""
    cols = st.columns(len(stats))
    for col, (value, label) in zip(cols, stats):
        with col:
            st.markdown(stat(value, label), unsafe_allow_html=True)


DEMAND_PATTERN_KIND = {
    "smooth": "good",
    "intermittent": "neutral",
    "erratic": "anomaly",
    "lumpy": "bad",
}


def grounding_pill(ratio: float) -> str:
    """Returns HTML for a status pill based on the grounding ratio (0-1)."""
    pct = ratio * 100
    if pct >= 90:
        kind, label = "good", "grounded"
    elif pct >= 70:
        kind, label = "warn", "partially grounded"
    else:
        kind, label = "bad", "low grounding"
    return (
        f'<span class="rc-pill rc-pill--{kind}">'
        f'<span class="rc-pill-dot"></span>{pct:.0f}% {label}</span>'
    )


def altair_theme(chart):
    """Applies the shared palette/typography to an Altair chart. Call right before
    st.altair_chart(chart, use_container_width=True)."""
    return (
        chart.properties(background=TOKENS["surface"])
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelFont="JetBrains Mono",
            labelFontSize=10.5,
            labelColor=TOKENS["text_muted"],
            titleFont="Inter",
            titleFontSize=11,
            titleColor=TOKENS["text_muted"],
            grid=True,
            gridColor=TOKENS["border"],
            domainColor=TOKENS["border"],
            tickColor=TOKENS["border"],
        )
        .configure_legend(
            labelFont="Inter",
            labelFontSize=11,
            labelColor=TOKENS["text_muted"],
            titleColor=TOKENS["text_muted"],
        )
        .configure_title(font="Space Grotesk", fontSize=13, color=TOKENS["text"])
    )
