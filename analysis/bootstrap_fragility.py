"""
Block bootstrap fragility — resample return sequences (diagnostic only).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.robustness_common import annualized_sharpe, cagr, max_drawdown_pct


def block_bootstrap(
    returns: pd.Series,
    spy_returns: pd.Series,
    *,
    block_size: int = 5,
    n_samples: int = 1000,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """
    Block bootstrap of portfolio returns (preserves short autocorrelation).
    Rebuilds equity from resampled returns starting at 1.0.
    """
    r = returns.dropna().to_numpy()
    spy = spy_returns.reindex(returns.index).dropna().to_numpy()
    n = len(r)
    if n < block_size + 10:
        empty = pd.DataFrame()
        return empty, {"error": "insufficient data"}

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    cagrs, sharpes, dds = [], [], []
    underperform, neg_sharpe = 0, 0

    spy_total = float(np.prod(1 + spy) - 1) if len(spy) == n else 0.0

    for _ in range(n_samples):
        idx_blocks = rng.integers(0, max(1, n - block_size + 1), size=n_blocks)
        sample = []
        for b in idx_blocks:
            sample.extend(r[b : b + block_size])
        sample = np.array(sample[:n])
        eq = np.cumprod(np.concatenate([[1.0], 1 + sample]))
        eq_s = pd.Series(eq)
        ret_s = pd.Series(sample)
        sh = annualized_sharpe(ret_s)
        cg = float(eq[-1] - 1.0)
        dd = max_drawdown_pct(eq_s)
        cagrs.append(cg)
        sharpes.append(sh)
        dds.append(dd)
        if sh < 0:
            neg_sharpe += 1
        if cg < spy_total:
            underperform += 1

    dist = pd.DataFrame({"cagr": cagrs, "sharpe": sharpes, "max_drawdown_pct": dds})
    summary = {
        "block_size": block_size,
        "n_samples": n_samples,
        "cagr_mean": float(dist["cagr"].mean()),
        "cagr_p5": float(dist["cagr"].quantile(0.05)),
        "cagr_p95": float(dist["cagr"].quantile(0.95)),
        "sharpe_mean": float(dist["sharpe"].mean()),
        "sharpe_p5": float(dist["sharpe"].quantile(0.05)),
        "prob_negative_sharpe": neg_sharpe / n_samples,
        "prob_underperform_spy": underperform / n_samples,
        "left_tail_loss_95": float(dist["cagr"].quantile(0.05)),
        "bootstrap_survival_rate": float((dist["sharpe"] > 0).mean()),
    }
    return dist, summary
