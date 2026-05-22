"""
Clarity score — rule-based thesis quality (not predictive).
"""

from __future__ import annotations

import re

from investment_app.memo.schema import CLARITY_LABELS


def _word_count(text: str) -> int:
    return len(re.findall(r"\w+", text or ""))


def _has_specifics(text: str) -> bool:
    t = (text or "").lower()
    if _word_count(text) < 12:
        return False
    markers = (
        "%", "quarter", "margin", "revenue", "rate", "capex", "consensus",
        "earnings", "if ", "when ", "unless", "because", "versus", "vs ",
    )
    return any(m in t for m in markers)


def score_memo_clarity(fields: dict) -> dict:
    """
    Dimensions ∈ [0, 1]:
    specificity, falsifiability, risk_awareness, horizon_clarity, driver_clarity
    """
    thesis = fields.get("thesis_summary", "")
    driver = fields.get("expected_driver", "")
    risks = fields.get("key_risks", "")
    inval = fields.get("invalidation_conditions", "")
    horizon = fields.get("time_horizon", "")
    mispricing = fields.get("market_mispricing", "")
    valuation = fields.get("valuation_case", "")
    why_now = fields.get("why_now", "")

    specificity = min(
        1.0,
        (
            (0.4 if _has_specifics(thesis) else 0.1)
            + (0.3 if _has_specifics(mispricing) else 0.0)
            + (0.3 if _has_specifics(valuation) else 0.0)
        ),
    )

    falsifiability = 0.1
    if _word_count(inval) >= 8:
        falsifiability += 0.4
    if any(w in inval.lower() for w in ("if ", "unless", "fail", "wrong", "below", "above")):
        falsifiability += 0.35
    if _word_count(inval) >= 20:
        falsifiability += 0.15
    falsifiability = min(1.0, falsifiability)

    risk_awareness = min(1.0, _word_count(risks) / 25.0) if risks else 0.0
    if "," in risks or ";" in risks or "\n" in risks:
        risk_awareness = min(1.0, risk_awareness + 0.2)

    horizon_clarity = 0.2
    h = horizon.lower().strip()
    if h in ("short", "medium", "long"):
        horizon_clarity = 0.85
    elif _word_count(horizon) >= 3:
        horizon_clarity = 0.7

    driver_clarity = 0.1
    if _word_count(driver) >= 15 and _has_specifics(driver):
        driver_clarity = 0.9
    elif _word_count(driver) >= 8:
        driver_clarity = 0.55

    why_now_score = min(1.0, _word_count(why_now) / 20.0) if why_now else 0.0

    breakdown = {
        "specificity": round(specificity, 2),
        "falsifiability": round(falsifiability, 2),
        "risk_awareness": round(risk_awareness, 2),
        "horizon_clarity": round(horizon_clarity, 2),
        "driver_clarity": round(driver_clarity, 2),
        "why_now_clarity": round(why_now_score, 2),
    }

    composite = (
        0.22 * breakdown["specificity"]
        + 0.22 * breakdown["falsifiability"]
        + 0.18 * breakdown["risk_awareness"]
        + 0.13 * breakdown["horizon_clarity"]
        + 0.20 * breakdown["driver_clarity"]
        + 0.05 * breakdown["why_now_clarity"]
    )

    if composite < 0.40:
        label = CLARITY_LABELS[0]  # WEAK THESIS
    elif composite < 0.68:
        label = CLARITY_LABELS[1]  # ACCEPTABLE
    else:
        label = CLARITY_LABELS[2]  # STRONGLY ARTICULATED

    return {
        "clarity_score": round(composite, 3),
        "clarity_label": label,
        "clarity_breakdown": breakdown,
    }
