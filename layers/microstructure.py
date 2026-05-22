"""
Market microstructure proxies — all causal (lag before rolling).
Consumed by execution_layer for spread/impact scaling.
"""

import numpy as np
import pandas as pd


def _lag(s: pd.Series, n: int = 1) -> pd.Series:
    return s.shift(n)


def build_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Temporal: feature[t] uses OHLCV <= t-1 only.
    """
    out = df.copy()
    c = _lag(out["Close"], 1)
    o = _lag(out["Open"], 1)
    h = _lag(out["High"], 1)
    l = _lag(out["Low"], 1)
    v = _lag(out["Volume"], 1)

    ret = c.pct_change()
    out["micro_spread_regime"] = ((h - l) / c.replace(0, np.nan)).rolling(20, min_periods=20).mean()
    out["micro_vol_regime"] = ret.rolling(20, min_periods=20).std() * np.sqrt(252)
    out["micro_liquidity_proxy"] = v.rolling(20, min_periods=20).mean()
    out["micro_overnight_gap"] = (o / c.shift(1).replace(0, np.nan) - 1.0).abs()
    out["micro_open_stress"] = (o - c.shift(1)).abs() / c.shift(1).replace(0, np.nan)
    hl = (h - l).replace(0, np.nan)
    out["micro_impact_proxy"] = out["micro_vol_regime"] * (1.0 / out["micro_liquidity_proxy"].replace(0, np.nan))
    return out


def execution_context(row: pd.Series) -> dict:
    return {
        "spread_bps": float(row.get("micro_spread_regime", 0.0005) or 0.0005) * 10000,
        "volatility": float(row.get("micro_vol_regime", 0.2) or 0.2),
        "volume": float(row.get("micro_liquidity_proxy", 1e6) or 1e6),
        "gap_risk": float(row.get("micro_overnight_gap", 0) or 0),
    }
