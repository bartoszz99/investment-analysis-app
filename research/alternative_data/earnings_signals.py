"""
Earnings surprise signals — aggregated from top holdings of SPY/QQQ.
Temporal: earnings assigned to announcement date, then shift(1) minimum.
LEAKAGE_RISK=HIGH when yfinance timing is ambiguous (after-hours unknown).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import yfinance as yf

from research.alternative_data.base import (
    AlternativeSignal,
    LeakageRisk,
    SignalMetadata,
    align_to_calendar,
    apply_lag,
)

# Representative holdings for earnings breadth (research proxy, not full index)
ETF_HOLDINGS_PROXY: dict[str, list[str]] = {
    "SPY": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "JPM", "XOM", "JNJ"],
    "QQQ": ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AVGO", "COST", "NFLX"],
}


def _fetch_earnings_surprises(ticker: str) -> pd.DataFrame:
    """
    Returns DataFrame index=date, columns=[surprise_pct, reported, estimate].
    yfinance timing: treat as known next session (lag applied upstream).
    """
    try:
        t = yf.Ticker(ticker)
        ed = t.earnings_dates
        if ed is None or (hasattr(ed, "empty") and ed.empty):
            return pd.DataFrame()
        df = ed.copy()
        if isinstance(df.index, pd.DatetimeIndex):
            df.index = df.index.tz_localize(None) if df.index.tz else df.index
        cols = df.columns.str.lower() if hasattr(df.columns, "str") else df.columns
        df.columns = [str(c).lower() for c in df.columns]

        surprise = pd.Series(dtype=float)
        for idx, row in df.iterrows():
            est = row.get("eps estimate") or row.get("estimate") or row.get("eps_estimate")
            rep = row.get("reported eps") or row.get("reported") or row.get("eps actual")
            if est is not None and rep is not None and not (pd.isna(est) or pd.isna(rep)):
                denom = abs(float(est)) if abs(float(est)) > 1e-6 else 1.0
                surprise.loc[idx] = (float(rep) - float(est)) / denom * 100.0
        if surprise.empty:
            return pd.DataFrame()
        return pd.DataFrame({"surprise_pct": surprise}).sort_index()
    except Exception:
        return pd.DataFrame()


def build_earnings_factors(
    calendar: pd.DatetimeIndex,
    etf_universe: tuple[str, ...] = ("SPY", "QQQ"),
    lag_days: int = 1,
) -> dict[str, AlternativeSignal]:
    """
    Aggregate earnings surprises to daily factors per ETF + universe mean.
    Lag policy: surprise on date D -> usable from session D+1 (shift 1).
    """
    daily: dict[str, pd.Series] = {}
    all_surprises: list[pd.Series] = []

    for etf in etf_universe:
        if etf not in ETF_HOLDINGS_PROXY:
            continue
        holdings = ETF_HOLDINGS_PROXY[etf]
        by_date: dict[pd.Timestamp, list[float]] = {}

        for stock in holdings:
            surp_df = _fetch_earnings_surprises(stock)
            for dt, row in surp_df.iterrows():
                d = pd.Timestamp(dt).normalize()
                by_date.setdefault(d, []).append(float(row["surprise_pct"]))

        if not by_date:
            mean_s = pd.Series(np.nan, index=calendar, name=f"earnings_mean_{etf}")
            breadth_s = pd.Series(np.nan, index=calendar, name=f"earnings_breadth_{etf}")
        else:
            mean_vals = {d: np.mean(v) for d, v in by_date.items()}
            breadth_vals = {d: np.mean([1 if x > 0 else 0 for x in v]) for d, v in by_date.items()}
            mean_raw = pd.Series(mean_vals).sort_index()
            breadth_raw = pd.Series(breadth_vals).sort_index()
            mean_aligned = align_to_calendar(mean_raw, calendar)
            breadth_aligned = align_to_calendar(breadth_raw, calendar)
            # Carry last known surprise until next event (research convention; MEDIUM risk)
            mean_s = apply_lag(mean_aligned.ffill(), lag_days)
            breadth_s = apply_lag(breadth_aligned.ffill(), lag_days)
            all_surprises.append(mean_s.dropna())

        daily[f"earnings_mean_{etf}"] = mean_s
        daily[f"earnings_breadth_{etf}"] = breadth_s

    # Universe aggregate
    if all_surprises:
        combined = pd.concat(all_surprises, axis=1).mean(axis=1)
        universe = apply_lag(align_to_calendar(combined, calendar), lag_days)
    else:
        universe = pd.Series(np.nan, index=calendar)
        warnings.warn("No earnings data retrieved; signals will be empty (HIGH leakage risk if used)")

    daily["daily_earnings_factor"] = universe

    meta_base = {
        "source": "yfinance earnings_dates (top holdings proxy)",
        "lag_days": lag_days,
        "update_frequency": "event-driven (quarterly per holding)",
        "leakage_risk": LeakageRisk.HIGH,
        "timestamp_assumption": "Earnings treated as known after market close on report date; "
        "usable from next session open only",
        "lag_policy": f"shift({lag_days}) after event date mapping; ffill between events",
    }

    signals: dict[str, AlternativeSignal] = {}
    for name, series in daily.items():
        signals[name] = AlternativeSignal(
            metadata=SignalMetadata(
                name=name,
                description=f"Earnings aggregation: {name}",
                **meta_base,
            ),
            series=series,
        )
    return signals
