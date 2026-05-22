"""
Hypothesis-first feature library — interpretable, low count.
Rule: shift(1) before all rolling stats; no global z-score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.equities.universe import DEFAULT_SECTOR, get_sector


def _lag(close: pd.DataFrame) -> pd.DataFrame:
    return close.shift(1)


# --- Hypothesis C: liquidity exhaustion (per-ticker wide) ---


def volume_zscore(volume: pd.Series, window: int = 20) -> pd.Series:
    lag_vol = volume.shift(1)
    mu = lag_vol.rolling(window, min_periods=window).mean()
    sigma = lag_vol.rolling(window, min_periods=window).std()
    return (lag_vol - mu) / sigma.replace(0, np.nan)


def extension_from_mean(close: pd.Series, window: int = 20) -> pd.Series:
    lag = close.shift(1)
    mu = lag.rolling(window, min_periods=window).mean()
    return lag / mu - 1.0


def failed_breakout(high: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    lag_h = high.shift(1)
    lag_c = close.shift(1)
    roll_max = lag_h.rolling(window, min_periods=window).max()
    broke = lag_h >= roll_max
    failed = broke & (close < close.shift(1))
    return failed.astype(float)


def gap_reversal(open_: pd.Series, close: pd.Series) -> pd.Series:
    gap = open_ / close.shift(1) - 1.0
    day_ret = close / open_ - 1.0
    return -gap * day_ret.shift(1)


def volatility_expansion(close: pd.Series, short: int = 5, long: int = 20) -> pd.Series:
    ret = close.shift(1).pct_change()
    vol_s = ret.rolling(short, min_periods=short).std()
    vol_l = ret.rolling(long, min_periods=long).std()
    return vol_s / vol_l.replace(0, np.nan)


def build_liquidity_exhaustion_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    c, h, o, v = ohlcv["Close"], ohlcv["High"], ohlcv["Open"], ohlcv["Volume"]
    return pd.DataFrame(
        {
            "liq_volume_zscore_20d": volume_zscore(v, 20),
            "liq_extension_20d": extension_from_mean(c, 20),
            "liq_failed_breakout_20d": failed_breakout(h, c, 20),
            "liq_gap_reversal": gap_reversal(o, c),
            "liq_vol_expansion_5_20": volatility_expansion(c, 5, 20),
        },
        index=ohlcv.index,
    )


def build_liquidity_panel(ohlcv: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {t: build_liquidity_exhaustion_features(df) for t, df in ohlcv.items()}


def liquidity_features_wide(ohlcv: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    panels = build_liquidity_panel(ohlcv)
    return {
        col: pd.concat({t: panels[t][col] for t in panels}, axis=1)
        for col in next(iter(panels.values())).columns
    }


# --- Hypothesis B: sector internal breadth ---


def _sector_groups(tickers: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for t in tickers:
        sec = get_sector(t)
        groups.setdefault(sec, []).append(t)
    return groups


def pct_above_ma(close: pd.DataFrame, window: int) -> pd.Series:
    lag = _lag(close)
    ma = lag.rolling(window, min_periods=window).mean()
    above = (lag > ma).astype(float)
    valid = ma.notna()
    return above.where(valid).mean(axis=1)


def pct_new_highs(close: pd.DataFrame, window: int = 20) -> pd.Series:
    lag = _lag(close)
    roll_max = lag.rolling(window, min_periods=window).max()
    at_high = (lag >= roll_max * 0.999).astype(float)
    return at_high.where(roll_max.notna()).mean(axis=1)


def return_dispersion(close: pd.DataFrame, window: int = 1) -> pd.Series:
    lag = _lag(close)
    ret = lag.pct_change(window)
    return ret.std(axis=1)


def equal_vs_cap_spread(close: pd.DataFrame) -> pd.Series:
    lag = _lag(close)
    ret = lag.pct_change(1)
    ew = ret.mean(axis=1)
    w = lag.div(lag.sum(axis=1), axis=0)
    cap = (ret * w).sum(axis=1)
    return ew - cap


def build_sector_breadth_for_group(close: pd.DataFrame) -> pd.DataFrame:
    """Sector-level breadth (one row per date)."""
    return pd.DataFrame(
        {
            "breadth_pct_above_ma20": pct_above_ma(close, 20),
            "breadth_pct_above_ma50": pct_above_ma(close, 50),
            "breadth_ew_cap_divergence": equal_vs_cap_spread(close),
            "breadth_new_high_participation": pct_new_highs(close, 20),
            "breadth_return_dispersion": return_dispersion(close, 1),
        },
        index=close.index,
    )


def map_sector_features_to_stocks(
    close_wide: pd.DataFrame,
    sector_features: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """
    Broadcast sector-level (lagged) features to each stock in that sector.
    Stocks in DEFAULT_SECTOR bucket use full-universe breadth as fallback.
    """
    groups = _sector_groups(list(close_wide.columns))
    full = build_sector_breadth_for_group(close_wide)
    full.columns = [f"breadth_universe_{c.replace('breadth_', '')}" for c in full.columns]

    sample = next(iter(sector_features.values()), build_sector_breadth_for_group(close_wide))
    all_feat_names = list(sample.columns)
    stock_frames: dict[str, dict[str, pd.Series]] = {t: {} for t in close_wide.columns}

    for sector, tickers in groups.items():
        if sector == DEFAULT_SECTOR:
            sf = full
        else:
            sub = close_wide[tickers]
            if len(tickers) < 3:
                sf = full
            else:
                sf = sector_features.get(sector, build_sector_breadth_for_group(sub))
        sf_lag = sf.shift(1)
        for t in tickers:
            for col in sf_lag.columns:
                stock_frames[t][col] = sf_lag[col]

    wide_out: dict[str, pd.DataFrame] = {}
    for col in all_feat_names:
        wide_out[col] = pd.concat(
            {t: stock_frames[t].get(col, pd.Series(np.nan, index=close_wide.index)) for t in close_wide.columns},
            axis=1,
        )
    return wide_out


def build_breadth_features_wide(close_wide: pd.DataFrame) -> dict[str, pd.DataFrame]:
    groups = _sector_groups(list(close_wide.columns))
    sector_features = {
        sec: build_sector_breadth_for_group(close_wide[tickers])
        for sec, tickers in groups.items()
        if len(tickers) >= 3 and sec != DEFAULT_SECTOR
    }
    return map_sector_features_to_stocks(close_wide, sector_features)


# --- Hypothesis A: post-earnings drift ---


def earnings_gap_on_event(open_: pd.Series, close: pd.Series, event_mask: pd.Series) -> pd.Series:
    gap = open_ / close.shift(1) - 1.0
    return gap.where(event_mask)


def abnormal_volume(volume: pd.Series, event_mask: pd.Series, window: int = 20) -> pd.Series:
    z = volume_zscore(volume, window)
    return z.where(event_mask)


def post_event_drift(close: pd.Series, event_mask: pd.Series, days: int) -> pd.Series:
    """
    Completed post-event drift, available only after the window finishes.
    Placed at event_end date, then shift(1) applied in build_earnings_features.
    """
    out = pd.Series(np.nan, index=close.index)
    event_idx = event_mask[event_mask].index
    for dt in event_idx:
        loc = close.index.get_loc(dt)
        if isinstance(loc, slice):
            loc = loc.start
        end = loc + days
        if end < len(close):
            seg = close.iloc[loc + 1 : end + 1]
            if len(seg) >= days:
                out.iloc[end] = seg.iloc[-1] / seg.iloc[0] - 1.0
    return out


def proxy_earnings_event_mask(ohlcv: pd.DataFrame) -> pd.Series:
    """
    Fallback when official calendar unavailable: large gap + elevated volume.
    PROXY ONLY — not exchange-timestamped (HIGH LEAKAGE RISK).
    """
    c, o, v = ohlcv["Close"], ohlcv["Open"], ohlcv["Volume"]
    gap = o / c.shift(1) - 1.0
    lag_gap = gap.shift(1)
    gap_std = lag_gap.rolling(20, min_periods=20).std()
    vol_z = volume_zscore(v, 20)
    return (lag_gap.abs() > 2.0 * gap_std) & (vol_z > 1.0)


def build_earnings_features(
    ohlcv: pd.DataFrame,
    event_dates: pd.DatetimeIndex | None,
    *,
    leakage_risk: str = "HIGH",
    use_proxy_if_empty: bool = True,
) -> pd.DataFrame:
    c, o, v = ohlcv["Close"], ohlcv["Open"], ohlcv["Volume"]
    if event_dates is None or len(event_dates) == 0:
        if use_proxy_if_empty:
            mask = proxy_earnings_event_mask(ohlcv)
            leakage_risk = "HIGH_PROXY_GAP_VOLUME"
        else:
            mask = pd.Series(False, index=ohlcv.index)
    else:
        ed = pd.DatetimeIndex(event_dates).normalize()
        idx = ohlcv.index.normalize()
        mask = pd.Series(idx.isin(ed), index=ohlcv.index)

    feats = pd.DataFrame(
        {
            "earn_earnings_gap": earnings_gap_on_event(o, c, mask),
            "earn_abnormal_volume": abnormal_volume(v, mask, 20),
            "earn_drift_persistence_3d": post_event_drift(c, mask, 3),
            "earn_drift_persistence_5d": post_event_drift(c, mask, 5),
            "earn_event_active": mask.astype(float),
        },
        index=ohlcv.index,
    )
    signal_cols = [c for c in feats.columns if c != "earn_event_active"]
    feats[signal_cols] = feats[signal_cols].shift(1)
    return feats.assign(_leakage_risk=leakage_risk)


def fetch_earnings_dates(ticker: str, *, cache: dict[str, pd.DatetimeIndex] | None = None) -> pd.DatetimeIndex:
    """
    yfinance earnings calendar — timestamp reliability limited.
    Mark HIGH LEAKAGE RISK in hypothesis report.
    """
    if cache is not None and ticker in cache:
        return cache[ticker]
    try:
        t = yf.Ticker(ticker)
        ed = t.earnings_dates
        if ed is None or (hasattr(ed, "empty") and ed.empty):
            result = pd.DatetimeIndex([])
        else:
            idx = ed.index
            if idx.tz is not None:
                idx = idx.tz_localize(None)
            result = pd.DatetimeIndex(idx.normalize().unique())
    except Exception:
        result = pd.DatetimeIndex([])
    if cache is not None:
        cache[ticker] = result
    return result


def build_earnings_panel(
    ohlcv: dict[str, pd.DataFrame],
    *,
    parallel: bool = True,
) -> tuple[dict[str, pd.DataFrame], dict]:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    meta: dict = {
        "leakage_risk": "HIGH",
        "timestamp_audit": {},
        "coverage": {},
        "proxy_fallback_used": {},
    }
    panels: dict[str, pd.DataFrame] = {}
    cache: dict[str, pd.DatetimeIndex] = {}

    def _dates_for(t: str) -> tuple[str, pd.DatetimeIndex]:
        return t, fetch_earnings_dates(t, cache=cache)

    earnings_by_ticker: dict[str, pd.DatetimeIndex] = {}
    tickers = list(ohlcv.keys())
    if parallel and len(tickers) > 1:
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(_dates_for, t): t for t in tickers}
            for fut in as_completed(futures):
                t, ed = fut.result()
                earnings_by_ticker[t] = ed
    else:
        for t in tickers:
            earnings_by_ticker[t] = fetch_earnings_dates(t, cache=cache)

    for t, df in ohlcv.items():
        ed = earnings_by_ticker.get(t, pd.DatetimeIndex([]))
        used_proxy = len(ed) == 0
        meta["timestamp_audit"][t] = {
            "n_events": len(ed),
            "source": "yfinance.earnings_dates" if not used_proxy else "proxy_gap_volume",
            "reliable": False,
        }
        meta["proxy_fallback_used"][t] = used_proxy
        meta["coverage"][t] = (
            float(proxy_earnings_event_mask(df).sum()) / max(len(df) / 252, 1)
            if used_proxy
            else len(ed) / max(len(df) / 252, 1)
        )
        feat = build_earnings_features(df, ed, leakage_risk="HIGH")
        panels[t] = feat.drop(columns=["_leakage_risk"], errors="ignore")
    return panels, meta
