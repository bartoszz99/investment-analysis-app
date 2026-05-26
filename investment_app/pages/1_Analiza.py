"""Analiza — ocena spójności pomysłu inwestycyjnego."""

from __future__ import annotations

import _path_setup  # noqa: F401

import investment_app.ui_common as ui

ui.setup_page()

import streamlit as st

from investment_app.i18n import translate

st.title(translate("page.analyze.title"))
st.caption(translate("page.analyze.caption"))

result = ui.run_analysis_form(save_journal=True)
if not result:
    ui.price_chart("SPY")
else:
    st.success(translate("page.analyze.saved_journal"))
    ui.render_analysis_result(result)
    c_w, c_m = st.columns(2)
    with c_w:
        if st.button(translate("page.analyze.add_watchlist"), use_container_width=True):
            from investment_app.watchlist.store import add_item

            add_item(
                result.ticker,
                asset_type=result.asset_type,
                idea_type=result.idea_type,
                horizon=result.horizon,
                market_region=result.market_region,
            )
            st.toast(translate("page.analyze.added_watchlist", ticker=result.ticker))
    with c_m:
        if st.button(translate("page.analyze.create_memo"), type="primary", use_container_width=True):
            from investment_app.memo.store import prefill_from_analysis

            st.session_state["memo_prefill"] = prefill_from_analysis(result)
            st.switch_page("pages/5_Notatki_inwestycyjne.py")
    if result.decision == "BUY":
        st.caption(translate("page.analyze.buy_hint"))
