"""
Neutralization variants — raw, sector, beta, momentum.
Trailing-only; no full-sample fits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.equities.universe import get_sector


def sector_neutral_wide(signal: pd.DataFrame, sectors: dict[str, str]) -> pd.DataFrame:
    """Cross-sectional demean within sector each date."""
    sec = pd.Series({k: sectors.get(k, "Other") for k in signal.columns})
    stacked = signal.stack(future_stack=True).reset_index()
    stacked.columns = ["date", "ticker", "value"]
    stacked["sector"] = stacked["ticker"].map(sec)
    stacked["value"] = stacked.groupby(["date", "sector"])["value"].transform(
        lambda x: x - x.mean() if len(x) >= 2 else x
    )
    out = stacked.pivot(index="date", columns="ticker", values="value")
    return out.reindex(index=signal.index, columns=signal.columns)


def _rolling_factor_residual(
    y: pd.Series,
    x: pd.Series,
    *,
    window: int = 60,
    min_periods: int = 20,
) -> pd.Series:
    """Trailing OLS residual vs single factor (vectorized rolling moments)."""
    mp = max(min_periods, window // 3)
    x_lag = x.shift(1)
    y_lag = y.shift(1)
    mean_x = x_lag.rolling(window, min_periods=mp).mean()
    mean_y = y_lag.rolling(window, min_periods=mp).mean()
    cov = (y_lag * x_lag).rolling(window, min_periods=mp).mean() - mean_y * mean_x
    var = x_lag.rolling(window, min_periods=mp).var()
    beta = cov / var.replace(0, np.nan)
    fitted = beta * x_lag
    return y - fitted


def beta_neutral_wide(
    signal: pd.DataFrame,
    spy_ret: pd.Series,
    *,
    window: int = 60,
) -> pd.DataFrame:
    """Per-ticker trailing residual vs SPY (vectorized)."""
    return pd.DataFrame(
        {_t: _rolling_factor_residual(signal[_t], spy_ret, window=window) for _t in signal.columns},
        index=signal.index,
    )


def momentum_neutral_wide(
    signal: pd.DataFrame,
    close: pd.DataFrame,
    *,
    window: int = 60,
    mom_lookback: int = 126,
    skip: int = 21,
) -> pd.DataFrame:
    """Per-ticker trailing residual vs 12-1 momentum proxy."""
    lag_close = close.shift(1)
    mom = lag_close / lag_close.shift(mom_lookback + skip) - 1.0
    mom = mom.shift(skip)
    cols = {}
    for t in signal.columns:
        if t in mom.columns:
            cols[t] = _rolling_factor_residual(signal[t], mom[t], window=window)
    return pd.DataFrame(cols, index=signal.index)


def apply_neutralization_suite(
    signal: pd.DataFrame,
    *,
    sectors: dict[str, str],
    spy_ret: pd.Series,
    close: pd.DataFrame,
    window: int = 60,
) -> dict[str, pd.DataFrame]:
    return {
        "raw": signal,
        "sector_neutral": sector_neutral_wide(signal, sectors),
        "beta_neutral": beta_neutral_wide(signal, spy_ret, window=window),
        "momentum_neutral": momentum_neutral_wide(signal, close, window=window),
    }


def classify_structural_exposure(
    ic_raw: float,
    ic_sector: float,
    ic_beta: float,
    ic_mom: float,
    *,
    collapse_threshold: float = 0.5,
) -> str:
    """If IC collapses after neutralization → structural exposure."""
    if np.isnan(ic_raw) or abs(ic_raw) < 0.01:
        return "no_signal"
    base = abs(ic_raw)
    if not np.isnan(ic_sector) and abs(ic_sector) < base * collapse_threshold:
        return "sector_exposure"
    if not np.isnan(ic_beta) and abs(ic_beta) < base * collapse_threshold:
        return "beta_exposure"
    if not np.isnan(ic_mom) and abs(ic_mom) < base * collapse_threshold:
        return "momentum_exposure"
    if (
        not np.isnan(ic_sector)
        and not np.isnan(ic_beta)
        and abs(ic_sector) >= base * collapse_threshold
        and abs(ic_beta) >= base * collapse_threshold
    ):
        return "potential_independent"
    return "mixed_or_weak"
