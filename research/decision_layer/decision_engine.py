"""
Final decision score and BUY / WATCH / IGNORE mapping.
"""

from __future__ import annotations

from dataclasses import dataclass

from research.decision_layer.counterparty_model import CounterpartyAssessment
from research.decision_layer.input_schema import AxisLabSnapshot, normalize_axis_score
from research.decision_layer.risk_filter import Decision, RiskFilterResult

W_STRUCTURAL = 0.4
W_TECHNICAL = 0.3
W_FUNDAMENTAL = 0.2
W_COUNTERPARTY = 0.1

BUY_THRESHOLD = 0.65
WATCH_THRESHOLD = 0.40


@dataclass
class DecisionResult:
    decision: Decision
    final_score: float
    breakdown: dict[str, float]
    classification: str


def compute_final_score(
    lab: AxisLabSnapshot,
    counterparty: CounterpartyAssessment,
) -> tuple[float, dict[str, float]]:
    breakdown = {
        "fundamental": normalize_axis_score(lab.score_fundamental),
        "technical": normalize_axis_score(lab.score_technical),
        "structural": normalize_axis_score(lab.score_structural),
        "counterparty": counterparty.counterparty_strength,
    }
    score = (
        W_STRUCTURAL * breakdown["structural"]
        + W_TECHNICAL * breakdown["technical"]
        + W_FUNDAMENTAL * breakdown["fundamental"]
        + W_COUNTERPARTY * breakdown["counterparty"]
    )
    return float(min(max(score, 0.0), 1.0)), breakdown


def score_to_decision(score: float) -> Decision:
    if score >= BUY_THRESHOLD:
        return Decision.BUY
    if score >= WATCH_THRESHOLD:
        return Decision.WATCH
    return Decision.IGNORE


def apply_cap(decision: Decision, cap: Decision | None) -> Decision:
    if cap is None:
        return decision
    order = {Decision.IGNORE: 0, Decision.WATCH: 1, Decision.BUY: 2}
    return decision if order[decision] <= order[cap] else cap


def decide(
    lab: AxisLabSnapshot,
    counterparty: CounterpartyAssessment,
    risk: RiskFilterResult,
) -> DecisionResult:
    if risk.forced is not None:
        final_score, breakdown = compute_final_score(lab, counterparty)
        return DecisionResult(
            decision=risk.forced,
            final_score=final_score,
            breakdown=breakdown,
            classification=lab.structural_class,
        )

    final_score, breakdown = compute_final_score(lab, counterparty)
    raw = score_to_decision(final_score)
    decision = apply_cap(raw, risk.cap)

    return DecisionResult(
        decision=decision,
        final_score=final_score,
        breakdown=breakdown,
        classification=lab.structural_class,
    )
