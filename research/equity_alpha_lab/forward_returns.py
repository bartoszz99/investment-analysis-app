"""
Forward returns — signal[t] predicts close-to-close from t+1 onward.
"""

from __future__ import annotations

import pandas as pd

from research.common.forward_returns import forward_return, forward_return_panel

HORIZONS = (1, 5, 10, 20)


def build_forward_returns_wide(close: pd.DataFrame) -> dict[int, pd.DataFrame]:
    return {h: forward_return_panel(close, h) for h in HORIZONS}


def apply_cost_haircut(spread: float, horizon: int, round_trip_bps: float = 10.0) -> float:
    """Simple cost sensitivity: subtract one-way cost scaled to horizon."""
    cost = round_trip_bps / 10_000.0
    return spread - cost if spread == spread else spread
