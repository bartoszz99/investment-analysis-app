"""
Hard rejection rules — cap decisions before scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from research.decision_layer.counterparty_model import CounterpartyAssessment
from research.decision_layer.input_schema import AxisLabSnapshot


class Decision(str, Enum):
    BUY = "BUY"
    WATCH = "WATCH"
    IGNORE = "IGNORE"


@dataclass
class RiskFilterResult:
    passed: bool
    cap: Decision | None  # max allowed decision if passed
    forced: Decision | None  # hard override
    reasons: list[str]


def apply_risk_filters(
    lab: AxisLabSnapshot,
    counterparty: CounterpartyAssessment,
) -> RiskFilterResult:
    reasons: list[str] = []
    forced: Decision | None = None
    cap: Decision | None = None

    if lab.structural_unit < 0.1:
        reasons.append(f"structural score {lab.structural_unit:.2f} < 0.10")
        forced = Decision.IGNORE

    if lab.neutralization_result == "MARKET_EXPOSURE" or lab.structural_class == "MARKET_EXPOSURE":
        reasons.append("neutralization: MARKET_EXPOSURE (beta/sector/momentum dominates)")
        forced = Decision.IGNORE

    if counterparty.counterparty_strength < 0.3:
        reasons.append(f"counterparty_strength {counterparty.counterparty_strength:.2f} < 0.30")
        forced = Decision.IGNORE

    if lab.regime_stability < 0.5:
        reasons.append(f"regime_stability {lab.regime_stability:.2f} < 0.50")
        cap = Decision.WATCH

    if forced == Decision.IGNORE:
        return RiskFilterResult(passed=False, cap=None, forced=Decision.IGNORE, reasons=reasons)

    return RiskFilterResult(passed=True, cap=cap, forced=None, reasons=reasons)
