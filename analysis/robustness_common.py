"""
Shared metrics for robustness / fragility analysis (diagnostic only).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def annualized_sharpe(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) < 2 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(252))


def cagr(equity: pd.Series) -> float:
    e = equity.dropna()
    if len(e) < 2 or e.iloc[0] <= 0:
        return 0.0
    years = len(e) / 252.0
    if years <= 0:
        return 0.0
    return float((e.iloc[-1] / e.iloc[0]) ** (1 / years) - 1)


def max_drawdown_pct(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity - peak) / peak.replace(0, np.nan)
    return float(abs(dd.min()) * 100) if len(dd) else 0.0


def hit_ratio(returns: pd.Series) -> float:
    r = returns.dropna()
    return float((r > 0).mean()) if len(r) else 0.0


def ols_alpha_beta(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> dict:
    aligned = pd.concat(
        [portfolio_returns.rename("p"), benchmark_returns.rename("b")],
        axis=1,
    ).dropna()
    if len(aligned) < 10:
        return {"alpha_daily": np.nan, "alpha_annualized": np.nan, "beta": np.nan}
    y = aligned["p"].to_numpy()
    x = aligned["b"].to_numpy()
    X = np.column_stack([np.ones(len(x)), x])
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return {
        "alpha_daily": float(coeffs[0]),
        "alpha_annualized": float(coeffs[0] * 252),
        "beta": float(coeffs[1]),
    }


def excess_return_vs_spy(strategy_equity: pd.Series, spy_close: pd.Series) -> float:
    s_norm = strategy_equity / strategy_equity.iloc[0]
    spy_eq = spy_close / spy_close.iloc[0]
    return float(s_norm.iloc[-1] - spy_eq.iloc[-1])


def fragility_score(sharpes: list[float]) -> float:
    arr = np.array([s for s in sharpes if s == s])
    if len(arr) < 2:
        return np.nan
    denom = abs(arr.mean())
    if denom < 1e-6:
        return float(arr.std())
    return float(arr.std() / denom)
