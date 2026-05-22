"""
Counterparty model — who is on the other side, and why are they forced to act?
"""

from __future__ import annotations

from dataclasses import dataclass

from research.decision_layer.input_schema import InvestmentIdea, TimeHorizon


@dataclass(frozen=True)
class CounterpartyAssessment:
    who: str
    forced_reason: str
    constraint: str
    counterparty_strength: float
    clarity: float
    narrative: str


_PROFILES: dict[str, dict] = {
    "momentum": {
        "who": "ETF rebalancers, CTAs, systematic trend followers",
        "forced_reason": "Mandate to follow price trend; vol-targeting and risk-parity flows",
        "constraint": "Capacity limits, crowding, reversal when positioning extreme",
        "strength": 0.75,
        "clarity": 0.85,
    },
    "breakout": {
        "who": "Stop-loss clusters, short-gamma dealers, breakout chasers",
        "forced_reason": "Technical triggers and hedging flows amplify break moves",
        "constraint": "False breaks when liquidity thin; dealer gamma flips",
        "strength": 0.55,
        "clarity": 0.65,
    },
    "earnings": {
        "who": "Algo reactors, retail repricers, slow institutional allocators",
        "forced_reason": "Mandate to adjust after information shock; reporting windows",
        "constraint": "Timestamps, spread widening, consensus already priced",
        "strength": 0.50,
        "clarity": 0.55,
    },
    "value": {
        "who": "Factor rebalancers, passive value-tilt funds, contrarian funds",
        "forced_reason": "Style mandate to buy cheap / sell rich on slow metrics",
        "constraint": "Value traps; macro regime dominates multiples",
        "strength": 0.45,
        "clarity": 0.60,
    },
    "macro": {
        "who": "Passive index allocators, pension rebalancers, risk-parity",
        "forced_reason": "Policy and macro shocks force asset-class rotation",
        "constraint": "Fully commoditized; edge is exposure not stock selection",
        "strength": 0.80,
        "clarity": 0.90,
    },
}


def assess_counterparty(idea: InvestmentIdea) -> CounterpartyAssessment:
    profile = _PROFILES.get(idea.signal_type, _PROFILES["momentum"])
    strength = profile["strength"]
    clarity = profile["clarity"]

    if idea.universe == "ETF":
        strength = min(strength + 0.1, 1.0)
        who = "Passive index flows, allocation mandates, ETF arbitrageurs"
        forced = "Index inclusion, pension policy weights, rebalance calendars"
    else:
        who = profile["who"]
        forced = profile["forced_reason"]

    # Horizon adjusts clarity (short = more microstructure noise)
    if idea.time_horizon == "short":
        clarity *= 0.85
    elif idea.time_horizon == "long":
        clarity *= 0.95

    narrative = (
        f"Other side: {who}. "
        f"Forced action: {forced}. "
        f"Constraint: {profile['constraint']}."
    )

    return CounterpartyAssessment(
        who=who,
        forced_reason=forced,
        constraint=profile["constraint"],
        counterparty_strength=float(min(max(strength, 0.0), 1.0)),
        clarity=float(min(max(clarity, 0.0), 1.0)),
        narrative=narrative,
    )
