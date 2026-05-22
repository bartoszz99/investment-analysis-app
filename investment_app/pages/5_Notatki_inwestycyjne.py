"""Notatki inwestycyjne — dyscyplina tezy przed poważnym przekonaniem."""

from __future__ import annotations

import investment_app.ui_common as ui

ui.setup_page()

import streamlit as st

from investment_app.data.market_region import REGIONS, parse_region
from investment_app.data.ticker_mapper import GPW_SEED_TICKERS, display_ticker, normalize_ticker, region_badge
from investment_app.i18n import (
    clarity_label,
    decision_label,
    horizon_label,
    region_label,
    review_label,
    translate,
)
from investment_app.memo.prompts import BUY_MEMO_REMINDER, GUIDANCE_BLOCKS
from investment_app.memo.review import REVIEW_OPTIONS, submit_review
from investment_app.memo.scoring import score_memo_clarity
from investment_app.memo.store import (
    ensure_examples_if_empty,
    get_memo,
    list_memos,
    prefill_from_analysis,
    save_memo,
)

_HORIZONS = ["short", "medium", "long"]

st.title(translate("page.memo.title"))
st.markdown(translate("page.memo.intro"))
st.info(BUY_MEMO_REMINDER)

ensure_examples_if_empty()

prefill = st.session_state.get("memo_prefill", {})
selected_id = st.session_state.get("memo_selected_id")

tab_create, tab_active, tab_review = st.tabs(
    [
        translate("page.memo.tab.create"),
        translate("page.memo.tab.active"),
        translate("page.memo.tab.review"),
    ]
)

with tab_create:
    with st.expander(translate("page.memo.guide_title"), expanded=False):
        for block in GUIDANCE_BLOCKS:
            st.markdown(f"**{block['field']}**")
            st.caption(translate("page.memo.weak", text=block["bad"]))
            st.markdown(translate("page.memo.strong", text=block["good"]))

    def _default(key: str, fallback: str = "") -> str:
        v = prefill.get(key)
        return v if v is not None and v != "" else fallback

    c1, c2 = st.columns(2)
    with c1:
        memo_region = st.selectbox(
            translate("ui.market_region"),
            REGIONS,
            format_func=region_label,
            index=REGIONS.index(prefill.get("market_region", "USA"))
            if prefill.get("market_region") in REGIONS
            else 0,
        )
        ticker = st.text_input(translate("page.memo.ticker_req"), value=_default("ticker", "AAPL")).upper()
        title = st.text_input(translate("page.memo.title_req"), value=_default("thesis_title", ""))
        summary = st.text_area(
            translate("page.memo.summary_req"),
            value=_default("thesis_summary", ""),
            height=100,
            placeholder=translate("page.memo.summary_ph"),
        )
        driver = st.text_area(
            translate("page.memo.driver_req"),
            value=_default("expected_driver", ""),
            height=80,
            placeholder=translate("page.memo.driver_ph"),
        )
        mispricing = st.text_area(
            translate("page.memo.mispricing"),
            value=_default("market_mispricing", ""),
            height=80,
        )
    with c2:
        risks = st.text_area(
            translate("page.memo.risks_req"),
            value=_default("key_risks", ""),
            height=100,
        )
        invalidation = st.text_area(
            translate("page.memo.invalidation_req"),
            value=_default("invalidation_conditions", ""),
            height=100,
            placeholder=translate("page.memo.invalidation_ph"),
        )
        why_now = st.text_area(translate("page.memo.why_now"), value=_default("why_now", ""), height=70)
        horizon = st.selectbox(
            translate("ui.time_horizon"),
            _HORIZONS,
            format_func=horizon_label,
            index=_HORIZONS.index(_default("time_horizon", "medium"))
            if _default("time_horizon", "medium") in _HORIZONS
            else 1,
        )
        valuation = st.text_area(
            translate("page.memo.valuation"),
            value=_default("valuation_case", ""),
            height=70,
        )
        confidence = st.slider(
            translate("page.memo.conviction"),
            0,
            100,
            int(prefill.get("confidence_0_100", 50)),
        )

    if prefill.get("linked_decision"):
        st.caption(
            translate(
                "page.memo.linked_analysis",
                decision=decision_label(prefill["linked_decision"]),
                score=f"{prefill.get('linked_analysis_score', 0):.0%}",
                summary=prefill.get("linked_explanation_summary", ""),
            )
        )

    live = score_memo_clarity(
        {
            "thesis_summary": summary,
            "expected_driver": driver,
            "key_risks": risks,
            "invalidation_conditions": invalidation,
            "time_horizon": horizon,
            "market_mispricing": mispricing,
            "valuation_case": valuation,
            "why_now": why_now,
        }
    )
    st.markdown(
        translate(
            "page.memo.clarity_draft",
            label=clarity_label(live["clarity_label"]),
            score=f"{live['clarity_score']:.0%}",
        )
    )
    bc = live["clarity_breakdown"]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(translate("page.memo.metric.specificity"), f"{bc['specificity']:.0%}")
    m2.metric(translate("page.memo.metric.falsifiable"), f"{bc['falsifiability']:.0%}")
    m3.metric(translate("page.memo.metric.risks"), f"{bc['risk_awareness']:.0%}")
    m4.metric(translate("page.memo.metric.horizon"), f"{bc['horizon_clarity']:.0%}")
    m5.metric(translate("page.memo.metric.driver"), f"{bc['driver_clarity']:.0%}")

    b1, b2 = st.columns(2)
    fields = {
        "ticker": ticker,
        "market_region": memo_region,
        "display_ticker": display_ticker(normalize_ticker(ticker, memo_region)),
        "thesis_title": title,
        "thesis_summary": summary,
        "expected_driver": driver,
        "market_mispricing": mispricing,
        "key_risks": risks,
        "invalidation_conditions": invalidation,
        "why_now": why_now,
        "time_horizon": horizon,
        "valuation_case": valuation,
        "confidence_0_100": confidence,
        "linked_analysis_score": prefill.get("linked_analysis_score"),
        "linked_decision": prefill.get("linked_decision"),
        "linked_explanation_summary": prefill.get("linked_explanation_summary"),
    }

    with b1:
        if st.button(translate("page.memo.save"), type="primary", use_container_width=True):
            if not ticker or not title or not summary or not invalidation:
                st.error(translate("page.memo.required_error"))
            else:
                m = save_memo(fields)
                st.success(translate("page.memo.saved", label=clarity_label(m.clarity_label)))
                st.session_state["memo_selected_id"] = m.id
                st.rerun()
    with b2:
        if st.button(translate("page.memo.save_clear"), use_container_width=True):
            st.session_state.pop("memo_prefill", None)
            st.rerun()

with tab_active:
    memos = list_memos()
    if not memos:
        st.info(translate("page.memo.no_memos"))
    else:
        for m in memos:
            created = (m.created_at or "")[:10]
            with st.container(border=True):
                h1, h2, h3 = st.columns([3, 2, 1])
                disp = m.display_ticker or display_ticker(m.ticker)
                h1.markdown(f"**{disp}** `{region_badge(m.market_region)}` — {m.thesis_title}")
                h1.caption(f"{created} · {clarity_label(m.clarity_label)}")
                dec = m.linked_decision or "—"
                dec_show = decision_label(dec) if dec in ("BUY", "WATCH", "IGNORE") else dec
                h2.markdown(
                    translate(
                        "page.memo.analysis_line",
                        decision=dec_show,
                        confidence=m.confidence_0_100,
                    )
                )
                if m.linked_analysis_score is not None:
                    h2.caption(
                        translate(
                            "page.memo.linked_score",
                            score=f"{m.linked_analysis_score:.0%}",
                        )
                    )
                status_label = review_label(m.review_status)
                h2.caption(translate("page.memo.review_status", status=status_label))
                if h3.button(translate("page.memo.open"), key=f"open_{m.id}"):
                    st.session_state["memo_selected_id"] = m.id
                    st.rerun()

    if selected_id:
        m = get_memo(selected_id)
        if m:
            st.divider()
            st.subheader(translate("page.memo.detail_title", ticker=m.ticker))
            st.markdown(f"### {m.thesis_title}")
            st.caption(
                translate(
                    "page.memo.clarity",
                    label=clarity_label(m.clarity_label),
                    score=f"{m.clarity_score:.0%}",
                    date=m.created_at[:10],
                )
            )
            st.markdown(f"**{translate('page.memo.summary')}**")
            st.write(m.thesis_summary)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**{translate('page.memo.expected_driver')}**")
                st.write(m.expected_driver)
                st.markdown(f"**{translate('page.memo.market_missing')}**")
                st.write(m.market_mispricing or "—")
                st.markdown(f"**{translate('page.memo.valuation_h')}**")
                st.write(m.valuation_case or "—")
            with col2:
                st.markdown(f"**{translate('page.memo.key_risks_h')}**")
                st.write(m.key_risks)
                st.markdown(f"**{translate('page.memo.invalidation_h')}**")
                st.write(m.invalidation_conditions)
                st.markdown(f"**{translate('page.memo.why_now_h')}**")
                st.write(m.why_now or "—")
            if m.linked_explanation_summary:
                st.caption(
                    translate(
                        "page.memo.linked_note",
                        text=m.linked_explanation_summary,
                    )
                )

with tab_review:
    memos = list_memos()
    if not memos:
        st.info(translate("page.memo.save_review_first"))
    else:
        memo_pick = st.selectbox(
            translate("page.memo.pick_review"),
            memos,
            format_func=lambda m: f"{m.ticker} — {m.thesis_title} ({m.id})",
        )
        if memo_pick:
            st.markdown(translate("page.memo.thesis_snip", text=memo_pick.thesis_summary[:300]))
            status = st.radio(
                translate("page.memo.how_thesis"),
                list(REVIEW_OPTIONS.keys()),
                format_func=review_label,
                horizontal=True,
            )
            right = st.text_area(
                translate("page.memo.what_right"),
                placeholder=translate("page.memo.what_right_ph"),
            )
            wrong = st.text_area(
                translate("page.memo.what_wrong"),
                placeholder=translate("page.memo.what_wrong_ph"),
            )
            market = st.text_area(
                translate("page.memo.market_cared"),
                placeholder=translate("page.memo.market_cared_ph"),
            )
            lessons = st.text_area(
                translate("page.memo.lessons"),
                value=memo_pick.lessons_learned,
                height=100,
            )
            notes = f"Prawidłowe: {right}\nBłędne: {wrong}\nRynek: {market}".strip()

            if memo_pick.market_region == "POLAND":
                st.markdown(f"**{translate('page.memo.gpw_review')}**")
                st.markdown(translate("page.memo.gpw_review_bullets"))

            if st.button(translate("page.memo.save_review"), type="primary"):
                submit_review(
                    memo_pick.id,
                    review_status=status,
                    review_notes=notes,
                    lessons_learned=lessons,
                )
                st.success(translate("page.memo.review_saved"))
                st.rerun()

            if memo_pick.review_notes:
                st.markdown(f"**{translate('page.memo.prev_review')}**")
                st.write(memo_pick.review_notes)
