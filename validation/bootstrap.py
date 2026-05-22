"""Bootstrap return distributions."""

import numpy as np
import pandas as pd


def bootstrap_sharpe_distribution(
    returns: pd.Series,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    r = returns.dropna().to_numpy()
    n = len(r)
    sharpes = []
    for _ in range(n_bootstrap):
        sample = r[rng.integers(0, n, size=n)]
        if sample.std() > 0:
            sharpes.append(sample.mean() / sample.std() * np.sqrt(252))
    arr = np.array(sharpes)
    return {
        "sharpe_p5": float(np.percentile(arr, 5)),
        "sharpe_p50": float(np.percentile(arr, 50)),
        "sharpe_p95": float(np.percentile(arr, 95)),
        "sharpe_std": float(arr.std()),
    }
