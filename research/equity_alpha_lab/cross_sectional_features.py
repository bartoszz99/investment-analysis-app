"""
Cross-sectional features — 3 hypothesis categories only.
All rolling stats use lagged prices (shift(1) before rolling).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf

from research.equity_alpha_lab.universe import DEFAULT_SECTOR, get_sector


def _lag_panel(close: pd.DataFrame) -> pd.DataFrame:
    return close.shift(1)


# ---------------------------------------------------------------------------
# C. Liquidity exhaustion / failed momentum
# ---------------------------------------------------------------------------


def _volume_zscore(vol: pd.Series, w: int = 20) -> pd.Series:
    lv = vol.shift(1)
    mu = lv.rolling(w, min_periods=w).mean()
    sd = lv.rolling(w, min_periods=w).std()
    return (lv - mu) / sd.replace(0, np.nan)


def _dollar_vol_zscore(close: pd.Series, vol: pd.Series, w: int = 20) -> pd.Series:
    dv = (close * vol).shift(1)
    mu = dv.rolling(w, min_periods=w).mean()
    sd = dv.rolling(w, min_periods=w).std()
    return (dv - mu) / sd.replace(0, np.nan)


def build_liquidity_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    c, h, o, v = ohlcv["Close"], ohlcv["High"], ohlcv["Open"], ohlcv["Volume"]
    lag_c = c.shift(1)
    roll_max = h.shift(1).rolling(20, min_periods=20).max()
    failed = ((h.shift(1) >= roll_max) & (c < c.shift(1))).astype(float)
    gap = o / c.shift(1) - 1.0
    ext5 = lag_c / lag_c.rolling(5, min_periods=5).mean() - 1.0
    ext10 = lag_c / lag_c.rolling(10, min_periods=10).mean() - 1.0
    rev = -gap.shift(1) * (c / o - 1.0).shift(1)
    return pd.DataFrame(
        {
            "liq_volume_zscore": _volume_zscore(v),
            "liq_dollar_volume_zscore": _dollar_vol_zscore(c, v),
            "liq_failed_breakout": failed,
            "liq_gap_extension": gap.shift(1),
            "liq_extension_5d": ext5,
            "liq_extension_10d": ext10,
            "liq_reversal_after_extreme": rev,
        },
        index=ohlcv.index,
    )


# ---------------------------------------------------------------------------
# B. Internal breadth / dispersion
# ---------------------------------------------------------------------------


def pct_above_sma(close: pd.DataFrame, w: int) -> pd.Series:
    lag = _lag_panel(close)
    ma = lag.rolling(w, min_periods=w).mean()
    return (lag > ma).where(ma.notna()).mean(axis=1)


def new_highs_vs_lows(close: pd.DataFrame, w: int = 20) -> pd.Series:
    lag = _lag_panel(close)
    hi = (lag >= lag.rolling(w, min_periods=w).max()).sum(axis=1)
    lo = (lag <= lag.rolling(w, min_periods=w).min()).sum(axis=1)
    return (hi - lo) / (hi + lo).replace(0, np.nan)


def cross_sectional_dispersion(close: pd.DataFrame) -> pd.Series:
    return _lag_panel(close).pct_change(1).std(axis=1)


def narrow_leadership_index(close: pd.DataFrame, top_n: int = 10) -> pd.Series:
    """Cap-top vs rest spread in daily returns (fragile leadership proxy)."""
    lag = _lag_panel(close)
    ret = lag.pct_change(1)
    top = lag.iloc[-1].nlargest(min(top_n, len(lag.columns))).index
    top_ret = ret[top].mean(axis=1)
    rest = ret.drop(columns=top, errors="ignore").mean(axis=1)
    return top_ret - rest


def sector_participation_breadth(close: pd.DataFrame) -> pd.Series:
    """Share of sectors with positive median return."""
    lag = _lag_panel(close)
    ret = lag.pct_change(1)
    sectors: dict[str, list[str]] = {}
    for t in close.columns:
        sectors.setdefault(get_sector(t), []).append(t)
    pos = []
    for dt in ret.index:
        n_pos = 0
        n_sec = 0
        for tickers in sectors.values():
            if len(tickers) < 2:
                continue
            med = ret.loc[dt, tickers].median()
            if med == med:
                n_sec += 1
                if med > 0:
                    n_pos += 1
        pos.append(n_pos / n_sec if n_sec else np.nan)
    return pd.Series(pos, index=ret.index)


def build_breadth_features(close: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "breadth_pct_above_sma50": pct_above_sma(close, 50),
        "breadth_pct_above_sma200": pct_above_sma(close, 200),
        "breadth_new_highs_lows": new_highs_vs_lows(close, 20),
        "breadth_cross_dispersion": cross_sectional_dispersion(close),
        "breadth_narrow_leadership": narrow_leadership_index(close),
        "breadth_sector_participation": sector_participation_breadth(close),
    }


def broadcast_market_features_to_stocks(
    market_feats: dict[str, pd.Series],
    tickers: list[str],
) -> dict[str, pd.DataFrame]:
    """Market-level breadth assigned to each stock (lagged), for cross-sectional IC."""
    out = {}
    for name, series in market_feats.items():
        lagged = series.shift(1)
        out[name] = pd.concat({t: lagged for t in tickers}, axis=1)
    return out


# ---------------------------------------------------------------------------
# A. Post-earnings underreaction
# ---------------------------------------------------------------------------


def proxy_earnings_mask(ohlcv: pd.DataFrame) -> pd.Series:
    """Gap + volume proxy when calendar missing — HIGH LEAKAGE RISK."""
    c, o, v = ohlcv["Close"], ohlcv["Open"], ohlcv["Volume"]
    gap = (o / c.shift(1) - 1.0).shift(1)
    gstd = gap.rolling(20, min_periods=20).std()
    vz = _volume_zscore(v)
    return (gap.abs() > 2 * gstd) & (vz > 1.0)


def fetch_earnings_dates(ticker: str, cache: dict) -> pd.DatetimeIndex:
    if ticker in cache:
        return cache[ticker]
    try:
        ed = yf.Ticker(ticker).earnings_dates
        if ed is None or ed.empty:
            cache[ticker] = pd.DatetimeIndex([])
        else:
            idx = ed.index
            if idx.tz is not None:
                idx = idx.tz_localize(None)
            cache[ticker] = pd.DatetimeIndex(idx.normalize().unique())
    except Exception:
        cache[ticker] = pd.DatetimeIndex([])
    return cache[ticker]


def build_earnings_features(
    ohlcv: pd.DataFrame,
    event_dates: pd.DatetimeIndex,
    *,
    use_proxy: bool,
) -> tuple[pd.DataFrame, str]:
    c, o, v = ohlcv["Close"], ohlcv["Open"], ohlcv["Volume"]
    if len(event_dates) == 0:
        mask = proxy_earnings_mask(ohlcv) if use_proxy else pd.Series(False, index=c.index)
        risk = "HIGH_LEAKAGE_PROXY" if use_proxy else "HIGH_LEAKAGE_NO_EVENTS"
    else:
        mask = pd.Series(c.index.normalize().isin(event_dates.normalize()), index=c.index)
        risk = "HIGH_LEAKAGE_CALENDAR"

    gap = (o / c.shift(1) - 1.0).where(mask)
    post_gap_cont = gap.shift(1)
    abvol = _volume_zscore(v).where(mask)

    drift = pd.Series(np.nan, index=c.index)
    for dt in mask[mask].index:
        loc = c.index.get_loc(dt)
        if isinstance(loc, slice):
            loc = loc.start
        end = loc + 5
        if end < len(c):
            seg = c.iloc[loc + 1 : end + 1]
            if len(seg) >= 5:
                drift.iloc[end] = seg.iloc[-1] / seg.iloc[0] - 1.0

    feats = pd.DataFrame(
        {
            "earn_post_gap_continuation": post_gap_cont,
            "earn_abnormal_volume": abvol,
            "earn_drift_surprise_5d": drift.shift(1),
            "earn_gap_size": gap.shift(1),
        },
        index=c.index,
    )
    return feats, risk


def build_earnings_panel(
    ohlcv: dict[str, pd.DataFrame],
    *,
    parallel: bool = True,
) -> tuple[dict[str, pd.DataFrame], dict]:
    cache: dict = {}
    dates: dict[str, pd.DatetimeIndex] = {}
    tickers = list(ohlcv.keys())

    if parallel and len(tickers) > 5:
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(fetch_earnings_dates, t, cache): t for t in tickers}
            for fut in as_completed(futs):
                t = futs[fut]
                dates[t] = fut.result()
    else:
        for t in tickers:
            dates[t] = fetch_earnings_dates(t, cache)

    panels, risks = {}, {}
    for t, df in ohlcv.items():
        use_proxy = len(dates[t]) == 0
        feat, risk = build_earnings_features(df, dates[t], use_proxy=use_proxy)
        panels[t] = feat
        risks[t] = risk

    return panels, {"per_ticker_risk": risks, "calendar_source": "yfinance_or_proxy"}


def build_all_features(
    close: pd.DataFrame,
    ohlcv: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], dict]:
    """
    Returns wide feature panels {feature_name: DataFrame(date x ticker)} and metadata.
    """
    tickers = list(close.columns)
    meta: dict = {"categories": {}}

    # C — per stock
    liq_panels = {t: build_liquidity_features(df) for t, df in ohlcv.items()}
    liq_wide = {
        col: pd.concat({t: liq_panels[t][col] for t in tickers}, axis=1)
        for col in liq_panels[tickers[0]].columns
    }

    # B — market broadcast to all stocks
    breadth = build_breadth_features(close)
    breadth_wide = broadcast_market_features_to_stocks(breadth, tickers)
    meta["categories"]["B_breadth"] = "market-level lagged broadcast"

    # A — per stock earnings
    earn_panels, earn_meta = build_earnings_panel(ohlcv)
    earn_wide = {
        col: pd.concat({t: earn_panels[t][col] for t in tickers}, axis=1)
        for col in earn_panels[tickers[0]].columns
    }
    meta["categories"]["A_earnings"] = earn_meta

    all_feats = {**liq_wide, **breadth_wide, **earn_wide}
    return all_feats, meta
