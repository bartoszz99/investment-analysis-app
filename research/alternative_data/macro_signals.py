"""
Macro regime signals — yfinance/FRED-style proxies (no FRED API required).
Temporal: all inputs lagged 1d before use; expanding z-scores only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

from research.alternative_data.base import (
    AlternativeSignal,
    LeakageRisk,
    SignalMetadata,
    align_to_calendar,
    apply_lag,
    expanding_zscore,
)

MACRO_TICKERS = {
    "rates": "^TNX",       # 10Y yield proxy
    "dollar": "UUP",       # dollar strength ETF
    "inflation": "TIP",    # inflation-linked bond ETF proxy
    "volatility": "^VIX",  # vol regime
}


def _fetch_close(ticker: str, period: str) -> pd.Series:
    hist = yf.Ticker(ticker).history(period=period)
    if hist.empty:
        return pd.Series(dtype=float)
    s = hist["Close"].copy()
    if s.index.tz:
        s.index = s.index.tz_localize(None)
    s.index = pd.DatetimeIndex([pd.Timestamp(d).normalize() for d in s.index])
    return s.sort_index()


def build_macro_signals(
    calendar: pd.DatetimeIndex,
    period: str = "1y",
    lag_days: int = 1,
) -> dict[str, AlternativeSignal]:
    """
    Build macro regime state and component factors.
    macro_regime_state: composite z-score of rates/dollar/inflation/vol changes.
    """
    from research.alternative_data.base import normalize_index

    calendar = normalize_index(calendar)
    components: dict[str, pd.Series] = {}
    for label, ticker in MACRO_TICKERS.items():
        close = _fetch_close(ticker, period)
        if close.empty:
            components[f"macro_{label}_change"] = pd.Series(np.nan, index=calendar)
            continue
        aligned = align_to_calendar(close, calendar)
        chg = apply_lag(aligned.pct_change(), lag_days)
        components[f"macro_{label}_change"] = chg

    # Composite regime: mean of expanding z-scores (causal)
    z_cols = []
    for label in MACRO_TICKERS:
        key = f"macro_{label}_change"
        z_cols.append(expanding_zscore(components[key]).rename(f"z_{label}"))
    z_df = pd.concat(z_cols, axis=1)
    regime_raw = z_df.mean(axis=1)
    components["macro_regime_state"] = regime_raw

    meta = {
        "source": "yfinance proxies (^TNX, UUP, TIP, ^VIX)",
        "lag_days": lag_days,
        "update_frequency": "daily",
        "leakage_risk": LeakageRisk.MEDIUM,
        "timestamp_assumption": "Macro prints assumed available after prior close; lag=1 session",
        "lag_policy": f"pct_change on lagged close, then shift({lag_days})",
    }

    signals: dict[str, AlternativeSignal] = {}
    for name, series in components.items():
        signals[name] = AlternativeSignal(
            metadata=SignalMetadata(name=name, description=f"Macro: {name}", **meta),
            series=series.reindex(calendar),
        )
    return signals
