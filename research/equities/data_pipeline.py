"""
Equity OHLCV pipeline — aligned calendar, liquidity filter, panel builders.
Anti-leakage: raw prices only; features lagged in feature_library.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

from research.equities.universe import UniverseSpec


@dataclass
class EquityPanel:
    close: pd.DataFrame
    volume: pd.DataFrame
    ohlcv: dict[str, pd.DataFrame]
    calendar: pd.DatetimeIndex
    dropped: list[str]


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    return df.sort_index()


def download_ticker(ticker: str, period: str = "3y") -> pd.DataFrame | None:
    try:
        raw = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    except Exception:
        return None
    if raw is None or raw.empty:
        return None
    df = _normalize_index(raw)
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    if "Close" not in cols:
        return None
    return df[cols]


def load_equity_panel(
    spec: UniverseSpec,
    *,
    period: str = "3y",
    min_avg_volume: float = 500_000,
    min_history_days: int = 252,
) -> EquityPanel:
    """Download universe; drop illiquid / short history names."""
    frames: dict[str, pd.DataFrame] = {}
    tickers = list(spec.tickers)
    try:
        bulk = yf.download(
            tickers,
            period=period,
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        if bulk is not None and not bulk.empty:
            if isinstance(bulk.columns, pd.MultiIndex):
                for t in tickers:
                    if t not in bulk.columns.get_level_values(0, dropna=False):
                        continue
                    sub = bulk[t].dropna(how="all")
                    if sub.empty or "Close" not in sub.columns:
                        continue
                    sub = _normalize_index(sub)
                    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in sub.columns]
                    if len(sub) >= min_history_days:
                        frames[t] = sub[cols]
            else:
                df = _normalize_index(bulk)
                if len(df) >= min_history_days and len(tickers) == 1:
                    frames[tickers[0]] = df[
                        [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
                    ]
    except Exception:
        pass

    for t in spec.tickers:
        if t in frames:
            continue
        df = download_ticker(t, period)
        if df is not None and len(df) >= min_history_days:
            frames[t] = df

    if not frames:
        raise RuntimeError("No equity data downloaded — check network or tickers")

    close_parts: list[pd.Series] = []
    vol_parts: list[pd.Series] = []
    dropped: list[str] = []

    for t, df in frames.items():
        avg_vol = df["Volume"].tail(60).mean()
        if avg_vol < min_avg_volume:
            dropped.append(t)
            continue
        close_parts.append(df["Close"].rename(t))
        vol_parts.append(df["Volume"].rename(t))

    if len(close_parts) < 20:
        raise RuntimeError(f"Too few liquid names after filter: {len(close_parts)}")

    close = pd.concat(close_parts, axis=1).sort_index().ffill()
    volume = pd.concat(vol_parts, axis=1).reindex(close.index).ffill()
    calendar = close.index

    ohlcv = {t: frames[t].reindex(calendar).ffill() for t in close.columns}
    return EquityPanel(
        close=close,
        volume=volume,
        ohlcv=ohlcv,
        calendar=calendar,
        dropped=dropped,
    )


def load_spy_benchmark(period: str = "3y") -> pd.Series:
    df = download_ticker("SPY", period)
    if df is None:
        raise RuntimeError("SPY benchmark download failed")
    return df["Close"]


def stack_wide(wide: pd.DataFrame, name: str = "value") -> pd.Series:
    """Wide (date x ticker) -> MultiIndex (date, ticker) Series."""
    s = wide.stack(future_stack=True)
    s.name = name
    return s
