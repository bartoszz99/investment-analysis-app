"""
Investment memo data model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


REVIEW_STATUSES = ("open", "thesis_playing_out", "unchanged", "weakening", "broken")
CLARITY_LABELS = ("WEAK THESIS", "ACCEPTABLE", "STRONGLY ARTICULATED")


@dataclass
class InvestmentMemo:
    id: str
    created_at: str
    ticker: str
    market_region: str = "USA"
    display_ticker: str = ""
    thesis_title: str
    thesis_summary: str
    expected_driver: str
    market_mispricing: str
    key_risks: str
    invalidation_conditions: str
    time_horizon: str
    valuation_case: str
    why_now: str
    confidence_0_100: int = 50
    linked_analysis_score: float | None = None
    linked_decision: str | None = None
    linked_explanation_summary: str | None = None
    review_status: str = "open"
    review_notes: str = ""
    lessons_learned: str = ""
    clarity_label: str = "ACCEPTABLE"
    clarity_score: float = 0.5
    clarity_breakdown: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InvestmentMemo:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        if not filtered.get("display_ticker"):
            from investment_app.data.ticker_mapper import display_ticker

            filtered["display_ticker"] = display_ticker(filtered.get("ticker", ""))
        filtered.setdefault("market_region", "USA")
        return cls(**filtered)
