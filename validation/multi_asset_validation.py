"""
Minimal multi-asset validation — descriptive only, not in signal path.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from validation.deflated_sharpe import deflated_sharpe_ratio


def asset_return_correlation(close_wide: pd.DataFrame) -> pd.DataFrame:
    """Correlation of daily close-to-close returns (descriptive)."""
    ret = close_wide.pct_change().dropna(how="all")
    return ret.corr()


def ranking_stability(ranks: pd.DataFrame, window: int = 20) -> dict:
    """
    Spearman-like stability: mean rank correlation between consecutive windows.
    Higher = more stable cross-sectional ordering.
    """
    if len(ranks) < window + 1:
        return {"mean_rank_corr": np.nan, "window": window}
    corrs = []
    vals = ranks.dropna(how="all")
    for i in range(window, len(vals)):
        a = vals.iloc[i - window].dropna().rank()
        b = vals.iloc[i].dropna().rank()
        common = a.index.intersection(b.index)
        if len(common) < 2:
            continue
        # Pearson on ranks == Spearman (no scipy dependency)
        corrs.append(a[common].rank().corr(b[common].rank()))
    return {
        "mean_rank_corr": float(np.nanmean(corrs)) if corrs else np.nan,
        "window": window,
        "n_obs": len(corrs),
    }


def turnover_report(turnover_daily: pd.Series) -> dict:
    td = turnover_daily.dropna()
    return {
        "mean_daily_turnover": float(td.mean()) if len(td) else 0.0,
        "total_turnover": float(td.sum()),
        "max_daily_turnover": float(td.max()) if len(td) else 0.0,
        "annualized_turnover": float(td.mean() * 252) if len(td) else 0.0,
    }


def portfolio_validation_summary(
    close_wide: pd.DataFrame,
    ranks: pd.DataFrame,
    turnover_daily: pd.Series,
    portfolio_returns: pd.Series,
    *,
    n_trials: int = 1,
) -> dict:
    return {
        "correlation_matrix": asset_return_correlation(close_wide).to_dict(),
        "ranking_stability": ranking_stability(ranks),
        "turnover": turnover_report(turnover_daily),
        "deflated_sharpe": deflated_sharpe_ratio(portfolio_returns, n_trials=n_trials),
    }


def save_turnover_report(report: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
