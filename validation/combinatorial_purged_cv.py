"""
Combinatorial Purged CV (CPCV) — simplified Lopez de Prado style.
Splits are time-ordered; embargo between train and test.
"""

import itertools
from typing import Iterator

import numpy as np
import pandas as pd


def generate_cpcv_splits(
    n_samples: int,
    n_test_groups: int = 2,
    embargo: int = 1,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    Yields (train_indices, test_indices) for combinatorial test group selections.
    """
    group_size = n_samples // n_test_groups
    groups = [np.arange(i * group_size, min((i + 1) * group_size, n_samples)) for i in range(n_test_groups)]
    for test_combo in itertools.combinations(range(n_test_groups), max(1, n_test_groups // 2)):
        test_idx = np.concatenate([groups[g] for g in test_combo])
        test_set = set(test_idx.tolist())
        train_mask = np.ones(n_samples, dtype=bool)
        for t in test_idx:
            for e in range(max(0, t - embargo), min(n_samples, t + embargo + 1)):
                train_mask[e] = False
        train_idx = np.where(train_mask)[0]
        train_idx = train_idx[~np.isin(train_idx, list(test_set))]
        yield train_idx, test_idx


def run_cpcv_backtest(
    returns: pd.Series,
    strategy_fn,
    n_groups: int = 4,
    embargo: int = 1,
) -> pd.DataFrame:
    """strategy_fn(train_returns) -> test_returns prediction or pnl series."""
    n = len(returns)
    rows = []
    for fold, (tr, te) in enumerate(generate_cpcv_splits(n, n_groups, embargo)):
        if len(te) == 0 or len(tr) < 5:
            continue
        pnl = strategy_fn(returns.iloc[tr], returns.iloc[te])
        rows.append({"fold": fold, "test_mean": float(pnl.mean()) if hasattr(pnl, "mean") else float(pnl)})
    return pd.DataFrame(rows)
