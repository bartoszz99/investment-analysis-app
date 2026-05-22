"""Wspólne elementy UI — USA i Polska (GPW)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from investment_app.data.market_region import REGIONS, parse_region
from investment_app.data.ticker_mapper import GPW_SEED_TICKERS, region_badge
from investment_app.engine import AnalysisRequest, AnalysisResult, analyze, fetch_prices
from investment_app.i18n import (
    asset_label,
    classification_label,
    decision_label,
    gauge_label as score_gauge_label,
    horizon_label,
    idea_label,
    region_label,
    risk_level_label,
    translate,
    translate_flag,
    trend_label,
    vol_label,
)

DECISION_COLORS = {"BUY": "#1a7f37", "WATCH": "#b8860b", "IGNORE": "#6b7280"}

_ASSET_TYPES = ["Stock", "ETF"]
_IDEAS = ["momentum", "value", "earnings", "breakout", "macro"]
_HORIZONS = ["short", "medium", "long"]


def setup_page(title: str | None = None) -> None:
    st.set_page_config(
        page_title=title or translate("app.title"),
        page_icon="📊",
        layout="wide",
    )


def region_selector_sidebar(*, key: str = "market_region") -> str:
    return st.sidebar.selectbox(
        translate("ui.market_region"),
        REGIONS,
        format_func=region_label,
        key=key,
    )


def render_decision_banner(decision: str, final_score: float, headline: str, region: str = "USA") -> None:
    badge = region_badge(region)
    color = DECISION_COLORS.get(decision, "#333")
    st.markdown(
        f"""
        <div style="background:{color};color:white;padding:1.2rem 1.5rem;
        border-radius:12px;margin-bottom:1rem;">
        <h2 style="margin:0;color:white;">{decision_label(decision)} <span style="font-size:0.6em;opacity:0.9;">{badge}</span></h2>
        <p style="margin:0.3rem 0 0 0;opacity:0.95;">
        {translate("ui.coherence_score")}: <b>{final_score:.0%}</b> — {headline}
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_gpw_diagnostics(result: AnalysisResult) -> None:
    st.subheader(translate("ui.gpw_diagnostics"))
    s = result.structural
    liq = s.get("liquidity", {})
    spec = s.get("speculation", {})
    narr = s.get("narrative", {})
    c1, c2, c3 = st.columns(3)
    c1.metric(
        translate("ui.liquidity_risk"),
        risk_level_label(liq.get("liquidity_risk")),
    )
    c2.metric(
        translate("ui.speculation_risk"),
        risk_level_label(spec.get("speculation_risk")),
    )
    c3.metric(
        translate("ui.narrative_dependence"),
        risk_level_label(narr.get("narrative_dependence")),
    )
    flags = s.get("ownership_flags", [])
    if flags:
        st.markdown(f"**{translate('ui.ownership_flags')}**")
        for f in flags:
            st.markdown(f"- {translate_flag(f)}")
    for w in result.explanation.get("gpw_warnings", []):
        st.warning(w)


def render_analysis_result(result: AnalysisResult, *, show_chart: bool = True) -> None:
    exp = result.explanation
    render_decision_banner(
        result.decision, result.final_score, exp["headline"], result.market_region
    )

    bd = result.breakdown
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        translate("ui.fundamental"),
        f"{bd['fundamental']:.0%}",
        score_gauge_label(bd["fundamental"]),
    )
    c2.metric(
        translate("ui.technical"),
        f"{bd['technical']:.0%}",
        score_gauge_label(bd["technical"]),
    )
    c3.metric(
        translate("ui.structural"),
        f"{bd['structural']:.0%}",
        classification_label(result.structural.get("classification")),
    )
    c4.metric(translate("ui.final_score"), f"{result.final_score:.0%}")

    if result.market_region == "POLAND":
        render_gpw_diagnostics(result)

    st.subheader(translate("ui.driving_idea"))
    st.write(exp["what_is_driving_this"])
    st.write(exp["likely_driver"])

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader(translate("ui.three_axis"))
        st.markdown(f"**{translate('ui.fundamentals_axis')}** — {exp['fundamental_blurb']}")
        st.caption(result.fundamental.get("detail", ""))
        st.markdown(f"**{translate('ui.technicals_axis')}** — {exp['technical_blurb']}")
        st.markdown(
            translate(
                "ui.trend_vol",
                trend=trend_label(result.technical.get("trend")),
                vol=vol_label(result.technical.get("vol_regime")),
            )
        )
        st.markdown(f"**{translate('ui.structure_axis')}** — {exp['structural_blurb']}")
    with col_b:
        st.subheader(translate("ui.counterparty"))
        st.write(exp["counterparty"])
        st.subheader(translate("ui.key_risks"))
        for r in exp["key_risks"]:
            st.markdown(f"- {r}")

    if show_chart:
        st.subheader(translate("ui.price_context"))
        price_chart(result.ticker, region=result.market_region)


def price_chart(ticker: str, *, region: str = "USA") -> None:
    try:
        df = fetch_prices(ticker, region=region)
        close = df["Close"]
        lag = close.shift(1)
        chart = pd.DataFrame(
            {
                "Cena": close,
                "SMA 50": lag.rolling(50, min_periods=50).mean(),
                "SMA 200": lag.rolling(200, min_periods=200).mean(),
            }
        ).dropna(how="all")
        st.line_chart(chart.tail(252), height=280)
    except Exception:
        st.caption(translate("ui.chart_unavailable"))


def run_analysis_form(
    *,
    default_ticker: str = "AAPL",
    default_region: str = "USA",
    save_journal: bool = True,
) -> AnalysisResult | None:
    region = region_selector_sidebar()
    with st.sidebar:
        st.header(translate("ui.your_idea"))
        if region == "POLAND":
            ex = st.selectbox(
                translate("ui.gpw_examples"),
                [""] + [t.replace(".WA", "") for t in GPW_SEED_TICKERS],
            )
            default_t = ex or "CDR"
        else:
            default_t = default_ticker
        ticker = st.text_input(translate("ui.ticker"), value=default_t).upper().strip()
        asset_type = st.selectbox(
            translate("ui.asset_type"),
            _ASSET_TYPES,
            format_func=asset_label,
        )
        idea_type = st.selectbox(
            translate("ui.idea_type"),
            _IDEAS,
            format_func=idea_label,
        )
        horizon = st.selectbox(
            translate("ui.time_horizon"),
            _HORIZONS,
            format_func=horizon_label,
        )
        user_note = st.text_area(
            translate("ui.user_note"),
            placeholder=translate("ui.user_note_ph"),
        )
        run = st.button(translate("ui.analyze_btn"), type="primary", use_container_width=True)

    if not run:
        return None
    if not ticker:
        st.error(translate("ui.enter_ticker"))
        return None

    with st.spinner(translate("ui.analyzing", ticker=ticker, region=region_label(region))):
        result = analyze(
            AnalysisRequest(
                ticker=ticker,
                asset_type=asset_type,
                idea_type=idea_type,
                horizon=horizon,
                market_region=region,
            ),
            user_note=user_note or None,
            save_journal=save_journal,
        )
    return result
