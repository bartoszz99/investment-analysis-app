"""
Portfolio construction v2 — rolling cov only, risk parity, vol target, constraints.
NO full-sample covariance estimation.
"""

import numpy as np
import pandas as pd


def _project_simplex_bounds(w: np.ndarray, max_weight: float) -> np.ndarray:
    w = np.clip(w, 0, max_weight)
    s = w.sum()
    return w / s if s > 0 else np.ones_like(w) / len(w)


def rolling_covariance(returns: pd.DataFrame, window: int = 60) -> np.ndarray:
    """Covariance using only trailing `window` rows."""
    tail = returns.tail(window).dropna()
    if len(tail) < 5:
        return np.eye(len(returns.columns))
    return tail.cov().to_numpy() * 252


def mean_variance_weights(
    expected_returns: np.ndarray,
    cov: np.ndarray,
    risk_aversion: float = 2.0,
    max_weight: float = 0.4,
) -> np.ndarray:
    n = len(expected_returns)
    if n <= 1:
        return np.ones(max(n, 1)) / max(n, 1)
    inv = np.linalg.pinv(cov + np.eye(n) * 1e-8)
    raw = inv @ expected_returns / max(risk_aversion, 1e-6)
    return _project_simplex_bounds(np.maximum(raw, 0), max_weight)


def risk_parity_weights(cov: np.ndarray, max_iter: int = 50) -> np.ndarray:
    n = cov.shape[0]
    w = np.ones(n) / n
    for _ in range(max_iter):
        port_var = w @ cov @ w
        if port_var <= 0:
            break
        mrc = cov @ w
        rc = w * mrc
        target = port_var / n
        rc_safe = np.where(rc > 0, rc, np.nan)
        w = w * (target / rc_safe)
        w = np.nan_to_num(w, nan=0.0)
        s = w.sum()
        w = w / s if s > 0 else np.ones(n) / n
    return _project_simplex_bounds(w, 0.5)


def volatility_target_leverage(realized_vol: float, target_vol: float = 0.15, max_lev: float = 2.0) -> float:
    if realized_vol <= 0 or np.isnan(realized_vol):
        return 1.0
    return float(np.clip(target_vol / realized_vol, 0.0, max_lev))


def optimize_portfolio(
    forecasts: pd.DataFrame,
    returns: pd.DataFrame,
    risk_aversion: float = 2.0,
    max_position: float = 0.4,
    target_vol: float = 0.15,
    turnover_penalty: float = 0.0,
    prev_weights: np.ndarray | None = None,
    method: str = "mean_variance",
    cov_window: int = 60,
    max_sector_exposure: float = 1.0,
) -> pd.Series:
    assets = list(forecasts.columns)
    mu = forecasts.iloc[-1].reindex(assets).fillna(0).to_numpy()
    ret = returns[assets].dropna()
    cov = rolling_covariance(ret, cov_window)

    if method == "risk_parity":
        w = risk_parity_weights(cov)
    else:
        w = mean_variance_weights(mu, cov, risk_aversion, max_position)

    if prev_weights is not None and turnover_penalty > 0 and len(prev_weights) == len(w):
        w = (1 - turnover_penalty) * w + turnover_penalty * prev_weights
        w = w / w.sum()

    w = _project_simplex_bounds(w, min(max_position, max_sector_exposure))
    vol = ret.iloc[-20:].std().mean() * np.sqrt(252) if len(ret) >= 20 else 0.2
    lev = volatility_target_leverage(vol, target_vol)
    return pd.Series(w * lev, index=assets)


def optimize_portfolio_v2(
    forecasts: pd.DataFrame,
    returns: pd.DataFrame,
    **kwargs,
) -> pd.Series:
    return optimize_portfolio(forecasts, returns, **kwargs)
