"""
Mandatory explanation for every decision — no output without rationale.
"""

from __future__ import annotations

from research.decision_layer.counterparty_model import CounterpartyAssessment
from research.decision_layer.decision_engine import DecisionResult
from research.decision_layer.input_schema import AxisLabSnapshot, InvestmentIdea
from research.decision_layer.risk_filter import RiskFilterResult


def build_explanation(
    idea: InvestmentIdea,
    lab: AxisLabSnapshot,
    counterparty: CounterpartyAssessment,
    risk: RiskFilterResult,
    result: DecisionResult,
) -> dict:
    structural_works = (
        lab.structural_class == "TRUE_SIGNAL"
        and lab.residual_ic_mean == lab.residual_ic_mean
        and abs(lab.residual_ic_mean) >= 0.05
    )

    if structural_works:
        struct_line = (
            f"Structural axis shows residual IC ({lab.residual_ic_mean:.3f}) "
            f"after neutralization — not pure beta proxy."
        )
    elif lab.structural_class == "MARKET_EXPOSURE":
        struct_line = (
            "Structural signal collapses after SPY/sector/momentum neutralization — "
            "this is market exposure, not independent mechanism."
        )
    else:
        struct_line = (
            f"Structural support weak (class={lab.structural_class}, "
            f"|IC|={abs(lab.ic_mean):.3f})."
        )

    exposure_type = _exposure_label(lab)
    killers = _what_kills_signal(lab, risk)

    reason = " ".join(
        [
            struct_line,
            counterparty.narrative,
            f"Exposure type: {exposure_type}.",
            f"Decision drivers: {'; '.join(killers) if killers else 'passed hard filters'}.",
        ]
    )

    return {
        "ticker": idea.ticker,
        "universe": idea.universe,
        "signal_type": idea.signal_type,
        "time_horizon": idea.time_horizon,
        "decision": result.decision.value,
        "final_score": round(result.final_score, 4),
        "breakdown": {k: round(v, 4) for k, v in result.breakdown.items()},
        "reason": reason,
        "classification": result.classification,
        "explanation": {
            "structural": struct_line,
            "counterparty": counterparty.narrative,
            "exposure_type": exposure_type,
            "signal_killers": killers,
            "independent_mechanism": structural_works,
        },
        "axis_raw": {
            "score_fundamental": lab.score_fundamental,
            "score_technical": lab.score_technical,
            "score_structural": lab.score_structural,
            "regime_stability": lab.regime_stability,
            "neutralization_result": lab.neutralization_result,
        },
        "counterparty": {
            "who": counterparty.who,
            "forced_reason": counterparty.forced_reason,
            "constraint": counterparty.constraint,
            "strength": counterparty.counterparty_strength,
            "clarity": counterparty.clarity,
        },
        "risk_filter_reasons": risk.reasons,
    }


def _exposure_label(lab: AxisLabSnapshot) -> str:
    if lab.structural_class == "TRUE_SIGNAL":
        return "possible_independent_structure (still not confirmed alpha)"
    if lab.structural_class == "MARKET_EXPOSURE":
        return "beta_sector_momentum_exposure"
    return "no_clear_exposure_or_noise"


def _what_kills_signal(lab: AxisLabSnapshot, risk: RiskFilterResult) -> list[str]:
    killers = list(risk.reasons)
    if lab.regime_stability < 0.5 and "regime" not in " ".join(killers):
        killers.append("unstable across bull/bear/vol regimes")
    if abs(lab.ic_mean) < 0.05:
        killers.append("IC below 0.05 threshold")
    return killers
