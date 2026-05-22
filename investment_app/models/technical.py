"""
Oś techniczna — trendy, momentum, zmienność, siła względem SPY.
Wszystkie cechy kroczące używają shift(1) przed rolling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _clip(x: float) -> float:
    return float(np.clip(x, -1.0, 1.0))


def analyze_technical(
    close: pd.Series,
    spy_close: pd.Series,
    idea: str = "momentum",
) -> dict:
    c = close.dropna()
    if len(c) < 220:
        return {
            "score_technical": 0.0,
            "trend": "unknown",
            "vol_regime": "unknown",
            "summary": "Za mało historii cenowej do wiarygodnej oceny technicznej.",
        }

    lag = c.shift(1)
    sma20 = lag.rolling(20, min_periods=20).mean()
    sma50 = lag.rolling(50, min_periods=50).mean()
    sma200 = lag.rolling(200, min_periods=200).mean()
    mom1 = lag / lag.shift(21) - 1.0
    mom3 = lag / lag.shift(63) - 1.0
    mom6 = lag / lag.shift(126) - 1.0
    vol20 = lag.pct_change().rolling(20, min_periods=20).std()

    spy = spy_close.reindex(c.index).ffill()
    rs = (lag / lag.shift(63)) / (spy.shift(1) / spy.shift(64)) - 1.0

    last = c.index[-1]
    px = lag.loc[last]
    parts: list[float] = []

    bull_stack = (
        sma50.loc[last] == sma50.loc[last]
        and sma200.loc[last] == sma200.loc[last]
        and px > sma50.loc[last] > sma200.loc[last]
    )
    bear_stack = (
        sma50.loc[last] == sma50.loc[last]
        and sma200.loc[last] == sma200.loc[last]
        and px < sma50.loc[last] < sma200.loc[last]
    )

    if bull_stack:
        parts.append(0.55)
        trend = "uptrend"
    elif bear_stack:
        parts.append(-0.55)
        trend = "downtrend"
    else:
        trend = "mixed"

    for m in (mom1, mom3, mom6):
        v = m.loc[last]
        if v == v:
            parts.append(_clip(v * 3.0))

    rv = rs.loc[last] if len(rs) else np.nan
    if rv == rv:
        parts.append(_clip(rv * 2.0))

    v20 = vol20.loc[last]
    if v20 == v20:
        if v20 > 0.04:
            parts.append(-0.25)
            vol_regime = "high"
        elif v20 < 0.015:
            vol_regime = "low"
            parts.append(0.05)
        else:
            vol_regime = "normal"
    else:
        vol_regime = "unknown"

    idea = idea.lower()
    if idea == "breakout":
        hi = lag.rolling(20, min_periods=20).max().loc[last]
        if hi == hi:
            parts.append(_clip((px / hi - 1.0) * 6.0))
    elif idea in ("value", "mean_reversion"):
        mu = sma20.loc[last]
        if mu == mu:
            parts.append(-_clip((px / mu - 1.0) * 4.0))

    score = _clip(float(np.mean(parts)) if parts else 0.0)

    if trend == "uptrend" and score > 0.2:
        summary = "Zachowanie ceny wspiera pomysł — trend i momentum się zgadzają."
    elif trend == "downtrend" and score < -0.2:
        summary = "Tło techniczne jest słabe dla byczej tezy."
    else:
        summary = "Technika jest mieszana — cena nie daje wyraźnej przewagi."

    return {
        "score_technical": score,
        "trend": trend,
        "vol_regime": vol_regime,
        "relative_strength_3m": float(rv) if rv == rv else None,
        "mom_3m": float(mom3.loc[last]) if mom3.loc[last] == mom3.loc[last] else None,
        "summary": summary,
    }
