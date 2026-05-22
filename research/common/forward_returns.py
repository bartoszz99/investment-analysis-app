"""
Forward return utilities — signal[t] predicts return from t+1 onward.
Anti-leakage: forward returns never overlap same-bar information.
"""

from __future__ import annotations

import pandas as pd


HORIZONS = (1, 5, 10, 20)


def forward_return(close: pd.Series, horizon: int) -> pd.Series:
    """Cumulative close-to-close return from t+1 through t+horizon."""
    return close.shift(-horizon) / close - 1.0


def forward_return_panel(close_wide: pd.DataFrame, horizon: int) -> pd.DataFrame:
    return close_wide.apply(lambda c: forward_return(c, horizon))


def build_forward_returns(
    close: pd.Series,
    horizons: tuple[int, ...] = HORIZONS,
) -> pd.DataFrame:
    return pd.DataFrame({f"fwd_{h}d": forward_return(close, h) for h in horizons})


def build_forward_returns_wide(
    close_wide: pd.DataFrame,
    horizons: tuple[int, ...] = HORIZONS,
) -> dict[int, pd.DataFrame]:
    return {h: forward_return_panel(close_wide, h) for h in horizons}
