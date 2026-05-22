"""
Passive-flow concentration features — SPY, QQQ, XLK.
LIMITATION: Uses current top-holdings universe as static proxy for all history.
Weights approximated by lagged price levels (not actual shares outstanding).
All features: shift(1) before rolling; expanding z-scores where noted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.breadth.breadth_features import pct_above_sma, return_dispersion
from research.breadth.universe_loader import ComponentPanel

PASSIVE_FLOW_ETFS = ("SPY", "QQQ", "XLK")

DATA_LIMITATION = (
    "Holdings are static current top-30 proxies applied to full history. "
    "Concentration levels are approximate; regime comparisons are directional only."
)


def _lag(close: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return close.shift(1)


def _cap_weight_proxy(lag_close: pd.DataFrame) -> pd.DataFrame:
    total = lag_close.sum(axis=1).replace(0, np.nan)
    return lag_close.div(total, axis=0)


def concentration_ratio(lag_close: pd.DataFrame, n: int) -> pd.Series:
    w = _cap_weight_proxy(lag_close)
    return w.apply(lambda row: row.nlargest(min(n, row.notna().sum())).sum(), axis=1)


def mega_cap_return_share(
    lag_close: pd.DataFrame,
    etf_return: pd.Series,
    n: int = 5,
) -> pd.Series:
    """Fraction of ETF daily return attributable to top-n cap-weighted components."""
    w = _cap_weight_proxy(lag_close)
    comp_ret = lag_close.pct_change(1)
    top_cols = list(lag_close.columns[: max(n, 1)])
    # dynamic top n by weight each day
    contrib = pd.Series(np.nan, index=lag_close.index)
    for dt in lag_close.index:
        row_w = w.loc[dt].dropna()
        if row_w.empty:
            continue
        tops = row_w.nlargest(min(n, len(row_w))).index
        contrib.loc[dt] = (comp_ret.loc[dt, tops] * row_w[tops]).sum()
    etf_r = etf_return.reindex(contrib.index)
    return contrib / etf_r.replace(0, np.nan)


def equal_weight_divergence(lag_close: pd.DataFrame, etf_return: pd.Series) -> pd.Series:
    ew_ret = lag_close.pct_change(1).mean(axis=1)
    return etf_return - ew_ret


def breadth_vs_price_divergence(
    etf_close: pd.Series,
    breadth: pd.Series,
    range_window: int = 60,
) -> pd.Series:
    lag_etf = _lag(etf_close)
    roll_min = lag_etf.rolling(range_window, min_periods=range_window).min()
    roll_max = lag_etf.rolling(range_window, min_periods=range_window).max()
    range_pct = (lag_etf - roll_min) / (roll_max - roll_min).replace(0, np.nan)
    breadth_z = (breadth - breadth.expanding(min_periods=60).mean()) / (
        breadth.expanding(min_periods=60).std().replace(0, np.nan)
    )
    return range_pct - breadth_z.reindex(range_pct.index)


def leadership_narrowing(breadth: pd.Series, window: int = 10) -> pd.Series:
    return -breadth.diff(window)


def expanding_zscore(series: pd.Series, min_periods: int = 60) -> pd.Series:
    lagged = series.shift(1)
    mu = lagged.expanding(min_periods=min_periods).mean()
    sigma = lagged.expanding(min_periods=min_periods).std()
    return (lagged - mu) / sigma.replace(0, np.nan)


def passive_flow_proxy(etf_return: pd.Series, median_comp_ret: pd.Series) -> pd.Series:
    """ETF strength despite weak median component performance."""
    etf_mom = etf_return.rolling(5, min_periods=5).sum()
    med_mom = median_comp_ret.rolling(5, min_periods=5).sum()
    return etf_mom - med_mom


def build_concentration_features(
    panel: ComponentPanel,
    etf_close: pd.Series,
) -> pd.DataFrame:
    lag_comp = _lag(panel.close)
    lag_etf = _lag(etf_close)
    etf_ret = lag_etf.pct_change(1)

    breadth = pct_above_sma(panel.close, 20)
    median_ret = lag_comp.pct_change(1).median(axis=1)
    conc5 = concentration_ratio(lag_comp, 5)
    conc10 = concentration_ratio(lag_comp, 10)

    out = pd.DataFrame(index=panel.close.index)
    out["concentration_ratio_top5"] = conc5
    out["concentration_ratio_top10"] = conc10
    out["mega_cap_return_share"] = mega_cap_return_share(lag_comp, etf_ret, n=5)
    out["equal_weight_divergence"] = equal_weight_divergence(lag_comp, etf_ret)
    out["breadth_pct_above_sma20"] = breadth
    out["breadth_vs_price_divergence"] = breadth_vs_price_divergence(etf_close, breadth)
    out["leadership_narrowing"] = leadership_narrowing(breadth)
    out["concentration_trend"] = expanding_zscore(conc10)
    out["dispersion_of_component_returns"] = return_dispersion(panel.close, 1)
    out["passive_flow_proxy"] = passive_flow_proxy(etf_ret, median_ret)
    out["etf_return_lagged"] = etf_ret
    return out
