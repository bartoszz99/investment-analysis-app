"""Monte Carlo robustness — detect fragile alpha."""

import numpy as np
import pandas as pd


def _sharpe(returns: np.ndarray) -> float:
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(252))


def monte_carlo_stability(
    returns: pd.Series,
    n_sims: int = 500,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    base = returns.dropna().to_numpy()
    base_sharpe = _sharpe(base)
    sims = []
    for _ in range(n_sims):
        shuffled = rng.permutation(base)
        sims.append(_sharpe(shuffled))
    sims = np.array(sims)
    stability = 1.0 - np.std(sims) / (abs(base_sharpe) + 1e-6)
    fragile = base_sharpe > 3.0 or stability < 0.3
    return {
        "base_sharpe": base_sharpe,
        "sim_sharpe_mean": float(sims.mean()),
        "sim_sharpe_std": float(sims.std()),
        "stability_score": float(np.clip(stability, 0, 1)),
        "fragile": fragile,
        "warning": "WARNING: likely overfit" if fragile else "OK",
    }
