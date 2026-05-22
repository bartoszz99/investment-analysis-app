"""
ETF component universe loader — public sources + static fallbacks.
US equities only; daily OHLCV via yfinance.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import pandas as pd
import yfinance as yf

# Static fallback: top holdings proxy (research-grade, manually curated)
STATIC_UNIVERSES: dict[str, list[str]] = {
    "SPY": [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "BRK-B", "JPM", "XOM", "JNJ",
        "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "KO", "PEP", "COST",
        "WMT", "MCD", "BAC", "CRM", "TMO", "CSCO", "ACN", "LIN", "ABT", "DHR",
    ],
    "QQQ": [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AVGO", "COST", "NFLX",
        "AMD", "ADBE", "PEP", "CSCO", "INTC", "CMCSA", "TMUS", "QCOM", "TXN", "AMAT",
        "HON", "INTU", "ISRG", "BKNG", "VRTX", "ADP", "GILD", "REGN", "LRCX", "MU",
    ],
    "XLK": [
        "AAPL", "MSFT", "NVDA", "AVGO", "CRM", "ADBE", "AMD", "ACN", "CSCO", "ORCL",
        "IBM", "INTU", "QCOM", "TXN", "AMAT", "NOW", "PANW", "ADI", "LRCX", "KLAC",
        "SNPS", "CDNS", "MCHP", "FTNT", "MSI", "APH", "NXPI", "HPQ", "KEYS", "IT",
    ],
    "XLF": [
        "BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "SPGI", "BLK",
        "C", "AXP", "CB", "PGR", "SCHW", "MMC", "ICE", "CME", "USB", "PNC",
        "AON", "TFC", "COF", "MET", "AIG", "BK", "AJG", "TRV", "ALL", "MSCI",
    ],
    "XLE": [
        "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "WMB", "OXY",
        "HES", "KMI", "HAL", "BKR", "DVN", "FANG", "TRGP", "OKE", "EQT", "APA",
        "CTRA", "MRO", "PXD", "CQP", "LNG", "TPL", "OVV", "PR", "AR", "RRC",
    ],
    "IWM": [
        "SMCI", "CELH", "FIX", "DUOL", "FN", "WGS", "MSTR", "CAR", "BOOT", "CVNA",
        "SMMT", "RGC", "PI", "FTAI", "GTLB", "ONON", "RBRK", "SEZL", "HIMS", "RKLB",
        "PLTR", "SOFI", "AFRM", "UPST", "RIVN", "LCID", "PATH", "AI", "IONQ", "RGTI",
    ],
}

ETF_TARGETS = tuple(STATIC_UNIVERSES.keys())


@dataclass
class ComponentPanel:
    etf: str
    tickers: list[str]
    close: pd.DataFrame  # index=dates, columns=components
    source: str


def _normalize_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return pd.DatetimeIndex([pd.Timestamp(d).normalize() for d in idx])


def _try_yfinance_holdings(etf: str, max_n: int = 40) -> list[str] | None:
    try:
        t = yf.Ticker(etf)
        # yfinance may expose funds_data.top_holdings on newer versions
        if hasattr(t, "funds_data"):
            fd = t.funds_data
            if hasattr(fd, "top_holdings") and fd.top_holdings is not None:
                df = fd.top_holdings
                if isinstance(df, pd.DataFrame) and len(df):
                    syms = [str(x).replace(".", "-") for x in df.index[:max_n]]
                    return syms
    except Exception:
        pass
    return None


def resolve_universe(etf: str) -> tuple[list[str], str]:
    live = _try_yfinance_holdings(etf)
    if live and len(live) >= 10:
        return live[:50], "yfinance_holdings"
    return STATIC_UNIVERSES[etf], "static_fallback"


def load_component_closes(
    etf: str,
    period: str = "2y",
    calendar: pd.DatetimeIndex | None = None,
) -> ComponentPanel:
    tickers, source = resolve_universe(etf)
    if not tickers:
        raise ValueError(f"No universe for {etf}")

    data = yf.download(
        tickers,
        period=period,
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    closes: dict[str, pd.Series] = {}
    if len(tickers) == 1:
        t = tickers[0]
        if "Close" in data.columns:
            closes[t] = data["Close"]
    else:
        for t in tickers:
            try:
                if t in data.columns.get_level_values(0):
                    closes[t] = data[t]["Close"]
            except (KeyError, TypeError):
                continue

    if not closes:
        warnings.warn(f"No component data for {etf}; using ETF only fallback")
        etf_px = yf.Ticker(etf).history(period=period)["Close"]
        closes = {etf: etf_px}

    close_df = pd.DataFrame(closes).sort_index()
    close_df.index = _normalize_index(close_df.index)
    close_df = close_df.dropna(how="all")

    if calendar is not None:
        cal = _normalize_index(calendar)
        close_df = close_df.reindex(cal)

    return ComponentPanel(etf=etf, tickers=list(close_df.columns), close=close_df, source=source)


def load_etf_close(etf: str, period: str = "2y") -> pd.Series:
    hist = yf.Ticker(etf).history(period=period)
    s = hist["Close"].copy()
    s.index = _normalize_index(s.index)
    return s.sort_index()
