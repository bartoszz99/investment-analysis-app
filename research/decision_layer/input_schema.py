"""
Investment idea input schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

UniverseType = Literal["ETF", "EQUITY"]
SignalType = Literal["momentum", "value", "earnings", "breakout", "macro"]
TimeHorizon = Literal["short", "medium", "long"]

SIGNAL_TO_LAB_IDEA: dict[str, str] = {
    "momentum": "momentum",
    "value": "mean_reversion",
    "earnings": "earnings_reaction",
    "breakout": "breakout",
    "macro": "momentum",
}


@dataclass(frozen=True)
class InvestmentIdea:
    ticker: str
    universe: UniverseType
    signal_type: SignalType
    time_horizon: TimeHorizon = "medium"

    def lab_idea_key(self) -> str:
        return SIGNAL_TO_LAB_IDEA.get(self.signal_type, "momentum")

    def key(self) -> tuple[str, str]:
        return (self.ticker.upper(), self.lab_idea_key())


@dataclass
class AxisLabSnapshot:
    """Loaded from 3-axis lab artifacts — decision layer does not recompute research."""

    score_fundamental: float
    score_technical: float
    score_structural: float
    structural_class: str
    neutralization_result: str
    ic_mean: float
    residual_ic_mean: float
    regime_stability: float
    verdict: str = ""

    @property
    def structural_unit(self) -> float:
        """Map structural axis [-1,1] to [0,1] magnitude for filters."""
        return float(min(max((self.score_structural + 1.0) / 2.0, 0.0), 1.0))


def normalize_axis_score(score: float) -> float:
    """[-1, 1] -> [0, 1] for decision weighting."""
    if score != score:
        return 0.5
    return float(min(max((score + 1.0) / 2.0, 0.0), 1.0))
