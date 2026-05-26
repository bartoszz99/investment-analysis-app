"""
Asystent analizy inwestycyjnej — punkt wejścia.

Uruchomienie:
    streamlit run investment_app/app.py

Strony w pasku bocznym: Analiza, Lista obserwowanych, Dziennik, Tezy, Notatki inwestycyjne.
"""

from __future__ import annotations

import path_setup  # noqa: F401

import investment_app.ui_common as ui

ui.setup_page()

import streamlit as st

from investment_app.i18n import translate

st.title(translate("app.title"))
st.markdown(translate("app.subtitle"))
st.markdown(
    f"""
    {translate("app.table_intro")}
    |------|---------|
    {translate("app.page.memo")}
    {translate("app.page.analyze")}
    {translate("app.page.watchlist")}
    {translate("app.page.journal")}
    {translate("app.page.thesis")}
    """
)
st.success(translate("app.success.start_memo"))
st.info(translate("app.info.analyze"))

