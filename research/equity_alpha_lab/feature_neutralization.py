"""
Early neutralization — trailing windows only, no full-sample fits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sector_neutral_wide(signal: pd.DataFrame, sectors: dict[str, str]) -> pd.DataFrame:
    stacked = signal.stack(future_stack=True).reset_index()
    stacked.columns = ["date", "ticker", "value"]
    sec = pd.Series(sectors)
    stacked["sector"] = stacked["ticker"].map(lambda t: sectors.get(t, "Other"))
    stacked["value"] = stacked.groupby(["date", "sector"])["value"].transform(
        lambda x: x - x.mean() if len(x) >= 2 else x
    )
    return stacked.pivot(index="date", columns="ticker", values="value").reindex(
        index=signal.index, columns=signal.columns
    )


def _rolling_residual(y: pd.Series, x: pd.Series, window: int = 60) -> pd.Series:
    mp = max(20, window // 3)
    x_lag, y_lag = x.shift(1), y.shift(1)
    mean_x = x_lag.rolling(window, min_periods=mp).mean()
    mean_y = y_lag.rolling(window, min_periods=mp).mean()
    cov = (y_lag * x_lag).rolling(window, min_periods=mp).mean() - mean_y * mean_x
    var = x_lag.rolling(window, min_periods=mp).var()
    beta = cov / var.replace(0, np.nan)
    return y - beta * x_lag


def beta_neutral_wide(signal: pd.DataFrame, spy_ret: pd.Series, window: int = 60) -> pd.DataFrame:
    return pd.DataFrame(
        {t: _rolling_residual(signal[t], spy_ret, window) for t in signal.columns},
        index=signal.index,
    )


def momentum_neutral_wide(
    signal: pd.DataFrame,
    close: pd.DataFrame,
    window: int = 60,
    lookback: int = 126,
    skip: int = 21,
) -> pd.DataFrame:
    lag = close.shift(1)
    mom = (lag / lag.shift(lookback + skip) - 1.0).shift(skip)
    return pd.DataFrame(
        {t: _rolling_residual(signal[t], mom[t], window) for t in signal.columns if t in mom.columns},
        index=signal.index,
    )


def neutralize_all(
    signal: pd.DataFrame,
    *,
    sectors: dict[str, str],
    spy_ret: pd.Series,
    close: pd.DataFrame,
    window: int = 60,
) -> dict[str, pd.DataFrame]:
    return {
        "raw": signal,
        "sector": sector_neutral_wide(signal, sectors),
        "beta": beta_neutral_wide(signal, spy_ret, window),
        "momentum": momentum_neutral_wide(signal, close, window),
    }
