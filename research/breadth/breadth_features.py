"""
ETF internal breadth features — component participation metrics.
Rule: lag component prices once globally; all rolling stats use lagged series.
Breadth[t] uses component data through t-1 only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.breadth.universe_loader import ComponentPanel


def _lag_components(close: pd.DataFrame) -> pd.DataFrame:
    return close.shift(1)


def pct_above_sma(close: pd.DataFrame, window: int) -> pd.Series:
    lag = _lag_components(close)
    sma = lag.rolling(window, min_periods=window).mean()
    above = (lag > sma).astype(float)
    valid = sma.notna()
    return above.where(valid).mean(axis=1)


def pct_new_high(close: pd.DataFrame, window: int) -> pd.Series:
    lag = _lag_components(close)
    roll_max = lag.rolling(window, min_periods=window).max()
    at_high = (lag >= roll_max).astype(float)
    valid = roll_max.notna()
    return at_high.where(valid).mean(axis=1)


def pct_positive_return(close: pd.DataFrame, window: int) -> pd.Series:
    lag = _lag_components(close)
    ret = lag.pct_change(window)
    pos = (ret > 0).astype(float)
    valid = ret.notna()
    return pos.where(valid).mean(axis=1)


def median_component_return(close: pd.DataFrame, window: int = 1) -> pd.Series:
    lag = _lag_components(close)
    ret = lag.pct_change(window)
    return ret.median(axis=1)


def return_dispersion(close: pd.DataFrame, window: int = 1) -> pd.Series:
    lag = _lag_components(close)
    ret = lag.pct_change(window)
    return ret.std(axis=1)


def advance_decline_ratio(close: pd.DataFrame) -> pd.Series:
    lag = _lag_components(close)
    ret = lag.pct_change(1)
    adv = (ret > 0).sum(axis=1)
    dec = (ret < 0).sum(axis=1)
    return adv / dec.replace(0, np.nan)


def equal_weight_return(close: pd.DataFrame) -> pd.Series:
    lag = _lag_components(close)
    return lag.pct_change(1).mean(axis=1)


def cap_weight_proxy_return(close: pd.DataFrame) -> pd.Series:
    """
    Cap-weight proxy: weight by lagged price level (rough size proxy without shares).
    Research approximation only.
    """
    lag = _lag_components(close)
    ret = lag.pct_change(1)
    w = lag.div(lag.sum(axis=1), axis=0)
    return (ret * w).sum(axis=1)


def equal_vs_cap_spread(close: pd.DataFrame) -> pd.Series:
    return equal_weight_return(close) - cap_weight_proxy_return(close)


def build_breadth_features(panel: ComponentPanel) -> pd.DataFrame:
    c = panel.close
    out = pd.DataFrame(index=c.index)
    out["pct_above_sma20"] = pct_above_sma(c, 20)
    out["pct_above_sma50"] = pct_above_sma(c, 50)
    out["pct_new_20d_high"] = pct_new_high(c, 20)
    out["pct_positive_5d_return"] = pct_positive_return(c, 5)
    out["median_component_return"] = median_component_return(c, 1)
    out["return_dispersion"] = return_dispersion(c, 1)
    out["equal_weight_vs_cap_weight"] = equal_vs_cap_spread(c)
    out["advance_decline_ratio"] = advance_decline_ratio(c)
    out["breadth_thrust"] = out["pct_above_sma20"].diff(5)
    out["breadth_divergence"] = out["pct_above_sma20"].diff(10)
    return out
