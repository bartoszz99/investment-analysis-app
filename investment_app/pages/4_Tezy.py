"""Tezy — przekonanie pisemne vs dowody."""

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

from dataclasses import asdict

from investment_app.engine import AnalysisRequest, analyze
from investment_app.i18n import translate
from investment_app.journal.store import load_entries
from investment_app.thesis.insights import generate_thesis_insights
from investment_app.thesis.store import add_thesis, delete_thesis, list_theses

st.title(translate("page.thesis.title"))
st.caption(translate("page.thesis.caption"))

with st.expander(translate("page.thesis.new_expander"), expanded=False):
    title = st.text_input(translate("page.thesis.short_title"))
    text = st.text_area(
        translate("page.thesis.my_thesis"),
        placeholder=translate("page.thesis.my_thesis_ph"),
        height=120,
    )
    tickers_raw = st.text_input(
        translate("page.thesis.linked_tickers"),
        placeholder=translate("page.thesis.linked_ph"),
    )
    if st.button(translate("page.thesis.save"), type="primary") and text.strip():
        tickers = [t.strip() for t in tickers_raw.split(",") if t.strip()]
        add_thesis(text, tickers, title=title or None)
        st.success(translate("page.thesis.saved"))
        st.rerun()

theses = list_theses()
if not theses:
    st.info(translate("page.thesis.empty"))
    st.stop()

thesis_ids = {f"{t['title']} ({t['id']})": t["id"] for t in the}
choice = st.selectbox(translate("page.thesis.select"), list(thesis_ids.keys()))
thesis = next(t for t in theses if t["id"] == thesis_ids[choice])

st.markdown(f"### {thesis['title']}")
st.write(thesis["text"])
tickers_str = ", ".join(thesis.get("tickers") or []) or translate("page.thesis.tickers_none")
st.caption(translate("page.thesis.tickers", tickers=tickers_str))

if st.button(translate("page.thesis.delete")):
    delete_thesis(thesis["id"])
    st.rerun()

st.divider()
st.subheader(translate("page.thesis.evidence"))

journal = load_entries()
journal_by_ticker = {}
for e in journal:
    journal_by_ticker.setdefault(e["ticker"], []).append(e)

for ticker in thesis.get("tickers") or []:
    with st.container(border=True):
        st.markdown(f"#### {ticker}")
        latest = journal_by_ticker.get(ticker, [None])[0] if journal_by_ticker.get(ticker) else None

        run_live = st.checkbox(
            translate("page.thesis.run_live", ticker=ticker),
            key=f"live_{ticker}",
        )
        analysis_dict = latest
        if run_live:
            with st.spinner(translate("page.thesis.analyzing")):
                r = analyze(
                    AnalysisRequest(ticker, "Stock", "momentum", "medium"),
                    save_journal=True,
                )
                analysis_dict = asdict(r)

        insights = generate_thesis_insights(thesis["text"], ticker, analysis_dict)

        col_s, col_c = st.columns(2)
        with col_s:
            st.markdown(f"**{translate('page.thesis.supporting')}**")
            for line in insights["supporting"]:
                st.markdown(f"- {line}")
        with col_c:
            st.markdown(f"**{translate('page.thesis.contradicting')}**")
            for line in insights["contradicting"]:
                st.markdown(f"- {line}")

        st.markdown(f"**{translate('page.thesis.structural_risks')}**")
        for line in insights["structural_risks"]:
            st.markdown(f"- {line}")
        if insights.get("gpw_structural_risks"):
            st.markdown(f"**{translate('page.thesis.gpw_risks')}**")
            for line in insights["gpw_structural_risks"]:
                st.markdown(f"- {line}")
        st.markdown(f"{translate('page.thesis.likely_driver')} {insights['likely_driver']}")
