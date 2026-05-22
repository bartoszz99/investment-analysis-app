"""Deflated Sharpe Ratio (Bailey & Lopez de Prado simplified, no scipy)."""

import math

import numpy as np
import pandas as pd


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def deflated_sharpe_ratio(
    returns: pd.Series,
    n_trials: int = 1,
    sharpe_benchmark: float = 0.0,
) -> dict:
    r = returns.dropna().to_numpy()
    if len(r) < 2:
        return {"dsr": 0.0, "sharpe": 0.0, "prob_overfit": 1.0}
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0
    n = len(r)
    skew = float(pd.Series(r).skew())
    kurt = float(pd.Series(r).kurtosis()) + 3
    sr_var = (1 + 0.5 * sharpe**2 - skew * sharpe + (kurt - 1) / 4 * sharpe**2) / max(n - 1, 1)
    sr_std = math.sqrt(max(sr_var, 1e-12))
    euler = 0.5772156649
    try:
        from statistics import NormalDist

        nd = NormalDist()
        max_z = (1 - euler) * nd.inv_cdf(1 - 1 / max(n_trials, 1)) + euler * nd.inv_cdf(
            1 - 1 / (max(n_trials, 1) * math.e)
        )
    except Exception:
        max_z = 1.96
    threshold = sharpe_benchmark + max_z * sr_std
    dsr = _norm_cdf((sharpe - threshold) / sr_std) if sr_std > 0 else 0.0
    return {
        "dsr": float(dsr),
        "sharpe": float(sharpe),
        "threshold": float(threshold),
        "prob_overfit": float(1 - dsr),
    }
