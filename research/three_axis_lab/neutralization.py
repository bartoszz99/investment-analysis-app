"""
Minimal trailing neutralization — R ~ SPY + sector + momentum (60d).
No full-sample fits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_ols_residual(
    y: pd.Series,
    factors: pd.DataFrame,
    window: int = 60,
) -> pd.Series:
    mp = max(20, window // 3)
    aligned = pd.concat([y.rename("y"), factors], axis=1).dropna()
    if aligned.empty:
        return pd.Series(dtype=float, index=y.index)

    cols = list(factors.columns)
    out = pd.Series(np.nan, index=aligned.index)
    yv = aligned["y"].to_numpy()
    xv = aligned[cols].to_numpy()

    for i in range(mp, len(aligned)):
        sl = slice(max(0, i - window), i)
        y_train = yv[sl]
        x_train = xv[sl]
        if len(y_train) < mp:
            continue
        X = np.column_stack([np.ones(len(y_train)), x_train])
        coeffs, _, _, _ = np.linalg.lstsq(X, y_train, rcond=None)
        out.iloc[i] = yv[i] - np.concatenate([[1.0], xv[i]]) @ coeffs

    return out.reindex(y.index)


def build_factor_panel(
    ticker: str,
    spy_ret: pd.Series,
    sector_ret: pd.Series | None,
    mom_proxy: pd.Series,
) -> pd.DataFrame:
    factors = pd.DataFrame({"spy": spy_ret, "momentum": mom_proxy})
    if sector_ret is not None:
        factors["sector"] = sector_ret
    return factors


def neutralize_signal(
    signal: pd.Series,
    factors: pd.DataFrame,
    window: int = 60,
) -> pd.Series:
    return rolling_ols_residual(signal, factors, window=window)
