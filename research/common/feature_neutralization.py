"""
Feature neutralization — trailing OLS residuals only (no full-sample fits).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_ols_residual(
    feature: pd.Series,
    factors: pd.DataFrame,
    window: int = 60,
    min_periods: int | None = None,
) -> pd.Series:
    """
    Residualize feature against factors using trailing window OLS at each t.
    Uses data strictly before t (features already lagged upstream).
    """
    min_periods = min_periods or max(20, window // 3)
    aligned = pd.concat([feature.rename("y"), factors], axis=1).dropna(how="any")
    if aligned.empty:
        return pd.Series(dtype=float, index=feature.index)

    cols = list(factors.columns)
    out = pd.Series(np.nan, index=aligned.index)
    y_all = aligned["y"].to_numpy()
    x_all = aligned[cols].to_numpy()

    for i in range(min_periods, len(aligned)):
        start = max(0, i - window)
        y = y_all[start:i]
        X = x_all[start:i]
        if len(y) < min_periods:
            continue
        Xd = np.column_stack([np.ones(len(y)), X])
        coeffs, _, _, _ = np.linalg.lstsq(Xd, y, rcond=None)
        x_t = np.concatenate([[1.0], x_all[i]])
        out.iloc[i] = y_all[i] - x_t @ coeffs

    return out.reindex(feature.index)


def rolling_zscore(series: pd.Series, window: int = 60) -> pd.Series:
    lagged = series.shift(1)
    mu = lagged.rolling(window, min_periods=max(20, window // 3)).mean()
    sigma = lagged.rolling(window, min_periods=max(20, window // 3)).std()
    return (lagged - mu) / sigma.replace(0, np.nan)


def cross_sectional_demean(wide: pd.DataFrame) -> pd.DataFrame:
    """Demean across columns per date (same-day cross-section only)."""
    return wide.sub(wide.mean(axis=1), axis=0)


def neutralize_feature_panel(
    feature: pd.Series,
    spy_ret: pd.Series,
    sector_rets: pd.DataFrame,
    momentum_proxy: pd.Series,
    window: int = 60,
) -> pd.Series:
    """Neutralize vs SPY + sectors + momentum (trailing OLS)."""
    factors = pd.concat(
        [
            spy_ret.rename("spy"),
            sector_rets,
            momentum_proxy.rename("mom"),
        ],
        axis=1,
    )
    return rolling_ols_residual(feature, factors, window=window)
