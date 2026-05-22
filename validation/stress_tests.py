"""Stress tests — perturb features and execution."""

import numpy as np
import pandas as pd

from validation.monte_carlo import _sharpe


def noisy_feature_perturbation(returns: pd.Series, noise_scale: float = 0.01, n: int = 50, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    base = _sharpe(returns.dropna().to_numpy())
    perturbed = []
    r = returns.dropna().to_numpy()
    for _ in range(n):
        noisy = r + rng.normal(0, noise_scale, size=len(r))
        perturbed.append(_sharpe(noisy))
    arr = np.array(perturbed)
    degraded = abs(arr.mean() - base) > 0.5 * abs(base + 1e-6)
    return {"base_sharpe": base, "perturbed_mean": float(arr.mean()), "degraded": degraded}


def execution_delay_perturbation(returns: pd.Series, lag: int = 1) -> dict:
    shifted = returns.shift(lag).dropna()
    return {
        "base_sharpe": _sharpe(returns.dropna().to_numpy()),
        "lagged_sharpe": _sharpe(shifted.to_numpy()),
    }


def run_stress_suite(returns: pd.Series) -> dict:
    from validation.bootstrap import bootstrap_sharpe_distribution
    from validation.monte_carlo import monte_carlo_stability

    return {
        "monte_carlo": monte_carlo_stability(returns),
        "bootstrap": bootstrap_sharpe_distribution(returns),
        "feature_noise": noisy_feature_perturbation(returns),
        "execution_lag": execution_delay_perturbation(returns),
    }
