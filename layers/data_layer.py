"""
Immutable OHLCV data layer — single- and multi-asset.
Temporal: alignment forward-fill is ONLY for calendar sync when explicitly flagged;
never used to impute features (features built per-ticker before alignment).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import yfinance as yf

REQUIRED_OHLCV = ("Open", "High", "Low", "Close", "Volume")


def fetch_raw(ticker: str, period: str = "1mo") -> pd.DataFrame:
    """Immutable OHLCV fetch — no feature transforms."""
    return yf.Ticker(ticker).history(period=period)


def validate_market_data(df: pd.DataFrame, ticker: str | None = None) -> None:
    label = f" ({ticker})" if ticker else ""
    if df.empty:
        raise ValueError(f"Empty market data{label}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"Index must be DatetimeIndex{label}")
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"Index must be sorted ascending{label}")
    if df.index.has_duplicates:
        raise ValueError(f"Duplicate timestamps in index{label}")
    missing = [c for c in REQUIRED_OHLCV if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns{label}: {missing}")
    if df[list(REQUIRED_OHLCV)].isna().any().any():
        n = int(df[list(REQUIRED_OHLCV)].isna().sum().sum())
        raise ValueError(f"NaN in required OHLCV{label}: {n} cells (clean before features)")


def align_and_clean(df: pd.DataFrame, ticker: str | None = None) -> pd.DataFrame:
    """Alignment / cleaning only — no features."""
    out = df.sort_index().dropna(subset=list(REQUIRED_OHLCV)).copy()
    validate_market_data(out, ticker)
    return out


def load_market_data(ticker: str, period: str = "1mo") -> pd.DataFrame:
    raw = fetch_raw(ticker, period)
    return align_and_clean(raw, ticker)


@dataclass
class MultiAssetOHLCV:
    """
    Per-ticker OHLCV panels on a synchronized calendar.
    Anti-leakage: panels hold raw prices only; features computed per ticker independently.
    """

    tickers: list[str]
    panels: dict[str, pd.DataFrame]
    calendar: pd.DatetimeIndex
    alignment: str = "outer_join"
    forward_filled: bool = False
    alignment_note: str = field(default_factory=str)

    def wide(self, field: str = "Close") -> pd.DataFrame:
        """Wide matrix: index=dates, columns=tickers."""
        return pd.DataFrame(
            {t: self.panels[t][field] for t in self.tickers},
            index=self.calendar,
        )

    def tradable_mask(self) -> pd.Series:
        """True on dates where every ticker has full OHLCV (no alignment fill)."""
        mask = pd.Series(True, index=self.calendar)
        for t in self.tickers:
            mask &= self.panels[t][list(REQUIRED_OHLCV)].notna().all(axis=1)
        return mask

    def drop_non_tradable(self) -> MultiAssetOHLCV:
        m = self.tradable_mask()
        idx = self.calendar[m]
        return MultiAssetOHLCV(
            tickers=self.tickers,
            panels={t: self.panels[t].loc[idx].copy() for t in self.tickers},
            calendar=idx,
            alignment=self.alignment,
            forward_filled=self.forward_filled,
            alignment_note=self.alignment_note,
        )


def load_multi_asset(
    tickers: list[str] | tuple[str, ...],
    period: str = "1y",
    *,
    forward_fill_alignment: bool = False,
) -> MultiAssetOHLCV:
    """
    Load and synchronize multiple tickers.
    forward_fill_alignment: if True, ffill OHLCV after outer join (calendar sync only).
    """
    panels: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        raw = fetch_raw(ticker, period)
        panels[ticker] = align_and_clean(raw, ticker)
    return synchronize_panels(
        panels,
        forward_fill=forward_fill_alignment,
        alignment_reason="market data alignment" if forward_fill_alignment else "",
    )


def synchronize_panels(
    panels: dict[str, pd.DataFrame],
    *,
    forward_fill: bool = False,
    alignment_reason: str = "",
) -> MultiAssetOHLCV:
    """
    Outer-join calendar across tickers.
    Default: no forward-fill (NaN where a ticker did not trade).
    """
    if not panels:
        raise ValueError("Empty panel dict")
    tickers = list(panels.keys())
    calendar = panels[tickers[0]].index
    for t in tickers[1:]:
        calendar = calendar.union(panels[t].index)
    calendar = calendar.sort_values()

    aligned: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        frame = panels[ticker].reindex(calendar)
        if forward_fill:
            frame[list(REQUIRED_OHLCV)] = frame[list(REQUIRED_OHLCV)].ffill()
        aligned[ticker] = frame

    note = alignment_reason if forward_fill else "outer_join_no_ffill"
    return MultiAssetOHLCV(
        tickers=tickers,
        panels=aligned,
        calendar=calendar,
        alignment="outer_join",
        forward_filled=forward_fill,
        alignment_note=note,
    )


def panels_to_multiindex(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack panels to MultiIndex columns (ticker, field)."""
    parts = []
    for ticker, df in panels.items():
        sub = df[list(REQUIRED_OHLCV)].copy()
        sub.columns = pd.MultiIndex.from_product([[ticker], sub.columns])
        parts.append(sub)
    return pd.concat(parts, axis=1).sort_index()
