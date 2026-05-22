"""
Final score and BUY / WATCH / IGNORE — region-aware weights.
"""

from __future__ import annotations

from enum import Enum

from investment_app.data.market_region import is_poland

W_STRUCTURAL_USA, W_TECHNICAL_USA, W_FUNDAMENTAL_USA = 0.4, 0.3, 0.3
W_STRUCTURAL_PL, W_TECHNICAL_PL, W_FUNDAMENTAL_PL = 0.30, 0.25, 0.45

BUY_AT = 0.65
WATCH_AT = 0.40

USA_STRONG = frozenset({"TRUE_SIGNAL", "STRUCTURAL_HYPOTHESIS"})
PL_STRONG = frozenset({"ACCEPTABLE", "STRUCTURAL_HYPOTHESIS"})
PL_IGNORE = frozenset({"FRAGILE", "NARRATIVE_DRIVEN", "LIQUIDITY_RISK"})


class Decision(str, Enum):
    BUY = "BUY"
    WATCH = "WATCH"
    IGNORE = "IGNORE"


def to_unit(score: float) -> float:
    if score != score:
        return 0.5
    return float(max(0.0, min(1.0, (score + 1.0) / 2.0)))


def structural_unit(score: float, classification: str, *, region: str) -> float:
    u = to_unit(score)
    if is_poland(region):
        if classification in PL_IGNORE:
            return min(u, 0.2)
        return u
    if classification == "MARKET_EXPOSURE":
        return min(u, 0.35)
    if classification == "NOISE":
        return 0.0
    return u


def compute_decision(
    score_fundamental: float,
    score_technical: float,
    score_structural: float,
    structural_class: str,
    *,
    regime_stability: float = 1.0,
    region: str = "USA",
    liquidity_risk: str | None = None,
    speculation_risk: str | None = None,
) -> dict:
    if is_poland(region):
        wf, wt, ws = W_FUNDAMENTAL_PL, W_TECHNICAL_PL, W_STRUCTURAL_PL
        strong = PL_STRONG
    else:
        wf, wt, ws = W_FUNDAMENTAL_USA, W_TECHNICAL_USA, W_STRUCTURAL_USA
        strong = USA_STRONG

    f, t, s = to_unit(score_fundamental), to_unit(score_technical), structural_unit(
        score_structural, structural_class, region=region
    )
    final_score = ws * s + wt * t + wf * f

    if is_poland(region):
        if structural_class in PL_IGNORE or liquidity_risk == "HIGH":
            decision = Decision.IGNORE
        elif final_score >= BUY_AT and structural_class in strong and speculation_risk != "HIGH":
            decision = Decision.BUY
        elif final_score >= WATCH_AT:
            decision = Decision.WATCH if speculation_risk != "HIGH" else Decision.WATCH
        else:
            decision = Decision.IGNORE
        if speculation_risk == "HIGH" and decision == Decision.BUY:
            decision = Decision.WATCH
    else:
        if structural_class == "MARKET_EXPOSURE":
            decision = Decision.IGNORE
        elif structural_class == "NOISE" and final_score < WATCH_AT:
            decision = Decision.IGNORE
        elif final_score >= BUY_AT and structural_class in strong:
            decision = Decision.BUY
        elif final_score >= WATCH_AT:
            decision = Decision.WATCH
        else:
            decision = Decision.IGNORE

    if regime_stability < 0.5 and decision == Decision.BUY:
        decision = Decision.WATCH

    return {
        "decision": decision.value,
        "final_score": round(final_score, 3),
        "breakdown": {"fundamental": round(f, 3), "technical": round(t, 3), "structural": round(s, 3)},
        "axis_scores": {
            "fundamental": score_fundamental,
            "technical": score_technical,
            "structural": score_structural,
        },
        "weights": {"fundamental": wf, "technical": wt, "structural": ws, "region": region},
    }
