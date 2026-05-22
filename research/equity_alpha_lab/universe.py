"""
S&P 500 universe loader — liquid equities only.

SURVIVORSHIP BIAS WARNING
-------------------------
This loader uses the *current* S&P 500 constituent list applied historically.
Delisted / removed names are not included. Backtests and IC studies therefore
overstate investability and may inflate signal strength. Treat all results as
upper-bound research estimates until a point-in-time membership file is added.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

SURVIVORSHIP_WARNING = (
    "Current S&P 500 membership applied to historical windows — "
    "survivorship bias present; delisted names excluded."
)

# Coarse GICS-style sectors for neutralization (extend as needed)
SECTOR_MAP: dict[str, str] = {
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology", "GOOGL": "Communication",
    "AMZN": "Consumer Discretionary", "META": "Communication", "BRK-B": "Financials",
    "JPM": "Financials", "V": "Financials", "UNH": "Health Care", "XOM": "Energy",
    "JNJ": "Health Care", "PG": "Consumer Staples", "MA": "Financials", "HD": "Consumer Discretionary",
    "CVX": "Energy", "MRK": "Health Care", "ABBV": "Health Care", "KO": "Consumer Staples",
    "PEP": "Consumer Staples", "COST": "Consumer Staples", "WMT": "Consumer Staples",
    "BAC": "Financials", "LLY": "Health Care", "TMO": "Health Care", "ORCL": "Technology",
    "DIS": "Communication", "NFLX": "Communication", "CRM": "Technology", "AMD": "Technology",
    "INTC": "Technology", "QCOM": "Technology", "TXN": "Technology", "AMAT": "Technology",
    "CAT": "Industrials", "GE": "Industrials", "HON": "Industrials", "UPS": "Industrials",
    "NEE": "Utilities", "SO": "Utilities", "DUK": "Utilities", "LIN": "Materials",
    "PM": "Consumer Staples", "RTX": "Industrials", "LOW": "Consumer Discretionary",
    "SPGI": "Financials", "GS": "Financials", "BLK": "Financials", "DE": "Industrials",
}
DEFAULT_SECTOR = "Other"

MIN_PRICE = 5.0
MIN_ADV_USD = 10_000_000.0
ADV_LOOKBACK = 60


@dataclass(frozen=True)
class UniverseSpec:
    name: str
    tickers: tuple[str, ...]
    survivorship_warning: str


@dataclass
class EquityPanel:
    close: pd.DataFrame
    volume: pd.DataFrame
    dollar_volume: pd.DataFrame
    ohlcv: dict[str, pd.DataFrame]
    calendar: pd.DatetimeIndex
    dropped: list[str]
    spec: UniverseSpec


def get_sector(ticker: str) -> str:
    return SECTOR_MAP.get(ticker.upper().replace(".", "-"), DEFAULT_SECTOR)


def load_sp500_tickers() -> list[str]:
    """Current S&P 500 symbols from Wikipedia; static fallback if fetch fails."""
    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            flavor="lxml",
        )
        syms = tables[0]["Symbol"].astype(str).str.replace(".", "-", regex=False)
        return syms.tolist()
    except Exception:
        pass
    # Fallback subset if offline
    return list(SECTOR_MAP.keys()) + [
        "AVGO", "ADBE", "CSCO", "ACN", "MCD", "ABT", "DHR", "VZ", "CMCSA", "IBM",
        "NOW", "INTU", "AMGN", "ISRG", "BKNG", "GILD", "ADP", "TJX", "SYK", "MDT",
        "PFE", "BMY", "COP", "EOG", "SLB", "MO", "SO", "DUK", "PLD", "REGN",
    ]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    return df.sort_index()


def load_universe(
    *,
    period: str = "3y",
    min_price: float = MIN_PRICE,
    min_adv_usd: float = MIN_ADV_USD,
    max_tickers: int | None = None,
) -> EquityPanel:
    tickers = load_sp500_tickers()
    if max_tickers:
        tickers = tickers[:max_tickers]

    frames: dict[str, pd.DataFrame] = {}
    try:
        bulk = yf.download(
            tickers,
            period=period,
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        if bulk is not None and not bulk.empty and isinstance(bulk.columns, pd.MultiIndex):
            for t in tickers:
                if t not in bulk.columns.get_level_values(0):
                    continue
                sub = _normalize(bulk[t].dropna(how="all"))
                if "Close" in sub.columns and len(sub) >= 252:
                    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in sub.columns]
                    frames[t] = sub[cols]
    except Exception:
        pass

    for t in tickers:
        if t in frames:
            continue
        try:
            raw = yf.Ticker(t).history(period=period, auto_adjust=True)
            if raw is not None and not raw.empty and len(raw) >= 252:
                raw = _normalize(raw)
                cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in raw.columns]
                frames[t] = raw[cols]
        except Exception:
            continue

    if not frames:
        raise RuntimeError("S&P 500 download failed — check network")

    dropped: list[str] = []
    close_parts, vol_parts = [], []
    for t, df in frames.items():
        adv = (df["Close"] * df["Volume"]).tail(ADV_LOOKBACK).mean()
        px = df["Close"].iloc[-1]
        if adv < min_adv_usd or px < min_price:
            dropped.append(t)
            continue
        close_parts.append(df["Close"].rename(t))
        vol_parts.append(df["Volume"].rename(t))

    close = pd.concat(close_parts, axis=1).sort_index().ffill()
    volume = pd.concat(vol_parts, axis=1).reindex(close.index).ffill()
    dollar_volume = close * volume
    calendar = close.index
    ohlcv = {t: frames[t].reindex(calendar).ffill() for t in close.columns}

    spec = UniverseSpec(
        name="sp500",
        tickers=tuple(close.columns),
        survivorship_warning=SURVIVORSHIP_WARNING,
    )
    return EquityPanel(
        close=close,
        volume=volume,
        dollar_volume=dollar_volume,
        ohlcv=ohlcv,
        calendar=calendar,
        dropped=dropped,
        spec=spec,
    )
