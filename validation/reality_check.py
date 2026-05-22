"""White's Reality Check — bootstrap vs benchmark (simplified)."""

import numpy as np
import pandas as pd


def whites_reality_check(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """
    H0: strategy does not outperform benchmark after accounting for multiple testing.
    Simplified: bootstrap excess return distribution.
    """
    rng = np.random.default_rng(seed)
    excess = (strategy_returns - benchmark_returns).dropna().to_numpy()
    if len(excess) < 10:
        return {"p_value": 1.0, "significant": False, "mean_excess": 0.0}
    observed = excess.mean()
    boot = []
    for _ in range(n_bootstrap):
        sample = excess[rng.integers(0, len(excess), size=len(excess))]
        boot.append(sample.mean())
    boot = np.array(boot)
    p_value = float((boot >= observed).mean())
    return {
        "p_value": p_value,
        "significant": p_value < 0.05,
        "mean_excess": float(observed),
        "boot_mean": float(boot.mean()),
    }
