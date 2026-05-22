"""Dziennik — historia rozumowania i kontrola."""

from __future__ import annotations

import investment_app.ui_common as ui

ui.setup_page()

import streamlit as st

from investment_app.data.ticker_mapper import display_ticker, region_badge
from investment_app.i18n import (
    classification_label,
    decision_label,
    idea_label,
    region_label,
    translate,
)
from investment_app.journal.followup import enrich_entry
from investment_app.journal.store import load_entries

st.title(translate("page.journal.title"))
st.caption(translate("page.journal.caption"))

entries = load_entries()
tickers = sorted({e.get("ticker", "") for e in entries if e.get("ticker")})
decisions_en = ["All", "BUY", "WATCH", "IGNORE"]
regions_en = ["All", "USA", "POLAND"]

f1, f2, f3, f4 = st.columns(4)
with f1:
    ft = st.selectbox(
        translate("page.journal.ticker_filter"),
        ["All"] + tickers,
        format_func=lambda x: translate("region.all") if x == "All" else x,
    )
with f2:
    fd = st.selectbox(
        translate("page.journal.decision_filter"),
        decisions_en,
        format_func=lambda d: translate("region.all") if d == "All" else decision_label(d),
    )
with f3:
    fr = st.selectbox(
        translate("page.journal.region_filter"),
        regions_en,
        format_func=lambda r: translate("region.all") if r == "All" else region_label(r),
    )
with f4:
    use_date = st.checkbox(translate("page.journal.filter_from_date"))
    fd_date = (
        st.date_input(translate("page.journal.from_date"), disabled=not use_date)
        if use_date
        else None
    )

date_from = fd_date.isoformat() if use_date and fd_date else None
filtered = load_entries(
    ticker=None if ft == "All" else ft,
    decision=fd,
    date_from=date_from,
)
if fr != "All":
    filtered = [e for e in filtered if e.get("market_region", "USA") == fr]

if not filtered:
    st.info(translate("page.journal.empty"))
    st.stop()

st.metric(translate("page.journal.entries_shown"), len(filtered))

for entry in filtered[:50]:
    enriched = enrich_entry(entry)
    with st.container(border=True):
        ts = (enriched.get("timestamp") or "")[:16].replace("T", " ")
        reg = region_badge(enriched.get("market_region", "USA"))
        disp = enriched.get("display_ticker") or display_ticker(enriched["ticker"])
        st.markdown(
            f"**{disp}** `{reg}` · {idea_label(enriched.get('idea_type', 'momentum'))} · "
            f"**{decision_label(enriched.get('decision'))}** · "
            f"{enriched.get('final_score', 0):.0%} · _{ts}_"
        )
        st.caption(
            translate(
                "page.journal.structure",
                cls=classification_label(enriched.get("structural_class")),
            )
        )

        c1, c2, c3 = st.columns(3)
        px0 = enriched.get("price_at_analysis")
        px1 = enriched.get("current_price")
        pct = enriched.get("pct_change_since_analysis")
        if px0:
            c1.metric(translate("page.journal.price_at"), f"${px0:.2f}")
        if px1:
            c2.metric(translate("page.journal.price_now"), f"${px1:.2f}")
        if pct is not None:
            c3.metric(translate("page.journal.change_since"), f"{pct:+.1f}%")

        if enriched.get("directionally_correct"):
            st.markdown(
                f"**{translate('page.journal.followup')}** {enriched['directionally_correct']}"
            )

        if enriched.get("headline"):
            st.write(enriched["headline"])
        if enriched.get("user_note"):
            st.markdown(f"{translate('page.journal.your_note')} {enriched['user_note']}")
        if enriched.get("what_is_driving"):
            st.caption(enriched["what_is_driving"])
