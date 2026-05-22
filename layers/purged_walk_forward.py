"""
Simplified purged walk-forward (Lopez de Prado style).
Train / embargo / test — no overlap of label information into test backtest.
"""

import pandas as pd

from layers.backtest_engine import START_CAPITAL, run_backtest
from strategies.base import BaseStrategy


def purged_walk_forward(
    df: pd.DataFrame,
    strategy: BaseStrategy,
    train_size: int,
    test_size: int,
    embargo: int = 0,
    start_capital: float = START_CAPITAL,
) -> pd.DataFrame:
    rows = []
    start = 0
    window_id = 0

    while start + train_size + embargo + test_size <= len(df):
        train_end = start + train_size
        test_start = train_end + embargo
        test_end = test_start + test_size

        window_df = df.iloc[start:test_end].copy()
        signaled = strategy.apply(window_df)
        test_df = signaled.iloc[test_start - start :].copy()

        metrics = run_backtest(test_df, start_capital)
        rows.append(
            {
                "window": window_id,
                "strategy": strategy.name,
                "train_end": df.index[train_end - 1],
                "test_start": df.index[test_start],
                "test_end": df.index[test_end - 1],
                "embargo_bars": embargo,
                "return_pct": metrics["return_pct"],
                "sharpe": metrics["sharpe"],
                "max_drawdown": metrics["max_drawdown"],
                "trades": metrics["trades"],
            }
        )
        start += test_size
        window_id += 1

    return pd.DataFrame(rows)


def calibration_test_split(
    df: pd.DataFrame, calibration_ratio: float = 0.7
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    split = max(1, int(n * calibration_ratio))
    return df.iloc[:split].copy(), df.iloc[split:].copy()
