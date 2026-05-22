"""
Liquidity exhaustion features — ETF-level OHLCV proxies.
All features: shift(1) before rolling; daily timeframe only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.common.feature_neutralization import rolling_zscore


def _lag(s: pd.Series) -> pd.Series:
    return s.shift(1)


def build_liquidity_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Required columns: Open, High, Low, Close, Volume.
    """
    o = ohlcv["Open"]
    h = ohlcv["High"]
    l = ohlcv["Low"]
    c = ohlcv["Close"]
    v = ohlcv["Volume"]

    lag_c = _lag(c)
    lag_v = _lag(v)
    lag_h = _lag(h)
    lag_l = _lag(l)
    lag_o = _lag(o)

    ret1 = lag_c.pct_change(1)
    ret3 = lag_c.pct_change(3)

    vol_mean = lag_v.rolling(20, min_periods=20).mean()
    vol_std = lag_v.rolling(20, min_periods=20).std()
    volume_zscore_20d = (lag_v - vol_mean) / vol_std.replace(0, np.nan)

    dollar_vol = lag_c * lag_v
    dv_mean = dollar_vol.rolling(20, min_periods=20).mean()
    dv_std = dollar_vol.rolling(20, min_periods=20).std()
    dollar_volume_zscore = (dollar_vol - dv_mean) / dv_std.replace(0, np.nan)

    sma20 = lag_c.rolling(20, min_periods=20).mean()
    distance_from_sma20 = (lag_c - sma20) / sma20.replace(0, np.nan)

    intraday_range = (lag_h - lag_l) / lag_c.replace(0, np.nan)
    range_mean = intraday_range.rolling(20, min_periods=20).mean()
    intraday_range_expansion = intraday_range / range_mean.replace(0, np.nan)

    realized_vol_10d = ret1.rolling(10, min_periods=10).std() * np.sqrt(252)
    vol_of_volume = lag_v.pct_change().rolling(20, min_periods=20).std()

    gap = lag_o / lag_c.shift(1).replace(0, np.nan) - 1.0
    gap_extension = gap.rolling(3, min_periods=3).sum()

    out = pd.DataFrame(
        {
            "volume_zscore_20d": volume_zscore_20d,
            "dollar_volume_zscore": dollar_volume_zscore,
            "return_3d_extension": ret3,
            "distance_from_sma20": distance_from_sma20,
            "intraday_range_expansion": intraday_range_expansion,
            "realized_volatility_10d": realized_vol_10d,
            "volatility_of_volume": vol_of_volume,
            "gap_extension": gap_extension,
        },
        index=ohlcv.index,
    )

    # Composite exhaustion: extension + volume spike
    out["exhaustion_composite"] = (
        rolling_zscore(out["return_3d_extension"].abs(), 60).fillna(0)
        + rolling_zscore(out["volume_zscore_20d"], 60).fillna(0)
        + rolling_zscore(out["distance_from_sma20"].abs(), 60).fillna(0)
    ) / 3.0

    return out
