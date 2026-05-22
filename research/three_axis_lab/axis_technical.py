"""
Technical axis — price-only diagnostics, all features lagged shift(1).
Score ∈ [-1, 1].
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _clip(x: float) -> float:
    return float(np.clip(x, -1.0, 1.0))


def score_technical(close: pd.Series, idea: str = "momentum") -> dict:
    c = close.dropna()
    if len(c) < 220:
        return {"score_technical": 0.0, "notes": "insufficient history"}

    lag = c.shift(1)
    sma20 = lag.rolling(20, min_periods=20).mean()
    sma50 = lag.rolling(50, min_periods=50).mean()
    sma200 = lag.rolling(200, min_periods=200).mean()

    mom1 = lag / lag.shift(21) - 1.0
    mom3 = lag / lag.shift(63) - 1.0
    mom6 = lag / lag.shift(126) - 1.0
    vol20 = lag.pct_change().rolling(20, min_periods=20).std()

    last = c.index[-1]
    px = lag.loc[last]
    parts: list[float] = []

    # SMA structure
    if sma50.loc[last] == sma50.loc[last] and sma200.loc[last] == sma200.loc[last]:
        if px > sma50.loc[last] > sma200.loc[last]:
            parts.append(0.6)
        elif px < sma50.loc[last] < sma200.loc[last]:
            parts.append(-0.6)

    if sma20.loc[last] == sma20.loc[last]:
        parts.append(_clip((px / sma20.loc[last] - 1.0) * 5.0))

    # Momentum stack
    for m in (mom1, mom3, mom6):
        v = m.loc[last]
        if v == v:
            parts.append(_clip(v * 3.0))

    # Volatility penalty (very high vol → lower conviction)
    v20 = vol20.loc[last]
    if v20 == v20 and v20 > 0.035:
        parts.append(-0.3)

    base = float(np.mean(parts)) if parts else 0.0

    # Idea tilt (shifts emphasis, not separate alpha)
    idea = idea.lower()
    if idea == "breakout":
        hi20 = lag.rolling(20, min_periods=20).max().loc[last]
        if hi20 == hi20:
            base += _clip((px / hi20 - 1.0) * 8.0)
    elif idea == "mean_reversion":
        mu = sma20.loc[last]
        if mu == mu:
            base -= _clip((px / mu - 1.0) * 5.0)
    elif idea == "earnings_reaction":
        gap = c.pct_change().iloc[-1]
        if gap == gap:
            base += _clip(gap * 10.0)

    return {
        "score_technical": _clip(base),
        "sma_aligned_bull": bool(
            px > sma50.loc[last] > sma200.loc[last]
            if sma50.loc[last] == sma50.loc[last]
            else False
        ),
        "mom_1m": float(mom1.loc[last]) if mom1.loc[last] == mom1.loc[last] else None,
        "mom_3m": float(mom3.loc[last]) if mom3.loc[last] == mom3.loc[last] else None,
        "mom_6m": float(mom6.loc[last]) if mom6.loc[last] == mom6.loc[last] else None,
        "vol_20d": float(v20) if v20 == v20 else None,
    }
