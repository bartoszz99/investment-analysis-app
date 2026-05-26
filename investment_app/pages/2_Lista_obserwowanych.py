"""Lista obserwowanych — monitorowanie spójności tezy w czasie."""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import investment_app.utils.path_setup  # noqa: F401

import investment_app.ui_common as ui

ui.setup_page()

import streamlit as st

from investment_app.data.ticker_mapper import display_ticker, region_badge
from investment_app.i18n import (
    asset_label,
    classification_label,
    decision_label,
    horizon_label,
    idea_label,
    translate,
    trend_label,
)
from investment_app.ui_common import region_selector_sidebar
from investment_app.watchlist.monitor import refresh_all
from investment_app.watchlist.store import add_item, list_items, remove_item

_ASSET_TYPES = ["Stock", "ETF"]
_IDEAS = ["momentum", "value", "earnings", "breakout", "macro"]
_HORIZONS = ["short", "medium", "long"]

st.title(translate("page.watchlist.title"))
st.caption(translate("page.watchlist.caption"))

region = region_selector_sidebar(key="wl_region")

col_add, col_refresh = st.columns([2, 1])
with col_add:
    nt = st.text_input(
        translate("page.watchlist.add_ticker"),
        placeholder=translate("page.watchlist.add_ph"),
    ).upper().strip()
    c1, c2, c3 = st.columns(3)
    with c1:
        at = st.selectbox(
            translate("page.watchlist.type"),
            _ASSET_TYPES,
            format_func=asset_label,
            key="wl_type",
        )
    with c2:
        it = st.selectbox(
            translate("page.watchlist.idea"),
            _IDEAS,
            format_func=idea_label,
            key="wl_idea",
        )
    with c3:
        hz = st.selectbox(
            translate("page.watchlist.horizon"),
            _HORIZONS,
            format_func=horizon_label,
            key="wl_hz",
        )
    if st.button(translate("page.watchlist.add_btn")) and nt:
        add_item(nt, asset_type=at, idea_type=it, horizon=hz, market_region=region)
        st.toast(translate("page.watchlist.added", ticker=nt))
        st.rerun()

with col_refresh:
    st.write("")
    st.write("")
    if st.button(translate("page.watchlist.refresh_all"), type="primary", use_container_width=True):
        with st.spinner(translate("page.watchlist.refreshing")):
            st.session_state["wl_refresh"] = refresh_all(save_journal=False)
        st.rerun()

items = list_items()
if not items:
    st.info(translate("page.watchlist.empty"))
    st.stop()

refreshed = st.session_state.get("wl_refresh")
if refreshed is None:
    st.caption(translate("page.watchlist.click_refresh"))
    display = items
else:
    by_ticker = {r["ticker"]: r for r in refreshed if "error" not in r}
    display = []
    for item in items:
        row = {**item, **by_ticker.get(item["ticker"], {})}
        display.append(row)

for row in display:
    ticker = row["ticker"]
    with st.container(border=True):
        h1, h2, h3 = st.columns([2, 2, 1])
        dec = row.get("last_decision") or "—"
        score = row.get("last_score")
        prev = row.get("prev_score")
        delta = row.get("score_delta")
        if delta is None and score is not None and prev is not None:
            delta = score - prev

        badge = region_badge(row.get("market_region", "USA"))
        h1.markdown(f"### {display_ticker(ticker)} `{badge}`")
        struct_disp = row.get("last_structural")
        if struct_disp:
            struct_disp = classification_label(struct_disp)
        else:
            struct_disp = translate("page.watchlist.not_refreshed")
        h1.caption(
            f"{idea_label(row.get('idea_type', 'momentum'))} · "
            f"{asset_label(row.get('asset_type', 'Stock'))} · {struct_disp}"
        )
        dec_label = decision_label(dec) if dec in ("BUY", "WATCH", "IGNORE") else dec
        h2.markdown(f"**{dec_label}**" + (f" · {score:.0%}" if score is not None else ""))
        if delta is not None:
            sign = "+" if delta >= 0 else ""
            h2.caption(translate("page.watchlist.score_change", delta=f"{sign}{delta:.0%}"))
        if row.get("last_trend"):
            h2.caption(
                translate("page.watchlist.trend", trend=trend_label(row["last_trend"]))
            )

        if h3.button(translate("page.watchlist.remove"), key=f"rm_{ticker}"):
            remove_item(ticker)
            st.rerun()

        for a in row.get("alerts") or []:
            st.warning(a)
        for f in row.get("risk_flags") or []:
            st.markdown(f"- ⚠ {f}")
