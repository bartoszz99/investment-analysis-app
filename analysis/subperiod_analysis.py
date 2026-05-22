"""
Subperiod analysis — edge stability across time windows (diagnostic only).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.robustness_common import (
    annualized_sharpe,
    cagr,
    hit_ratio,
    max_drawdown_pct,
    ols_alpha_beta,
)


def _window_metrics(
    equity: pd.Series,
    turnover: pd.Series,
    spy_close: pd.Series,
    label: str,
) -> dict:
    ret = equity.pct_change().dropna()
    spy_ret = spy_close.pct_change().reindex(ret.index)
    ab = ols_alpha_beta(ret, spy_ret)
    return {
        "period": label,
        "start": str(equity.index.min().date()) if len(equity) else None,
        "end": str(equity.index.max().date()) if len(equity) else None,
        "n_days": len(equity),
        "sharpe": annualized_sharpe(ret),
        "cagr": cagr(equity),
        "max_drawdown_pct": max_drawdown_pct(equity),
        "turnover_annual": float(turnover.reindex(equity.index).mean() * 252),
        "beta": ab["beta"],
        "residual_alpha_annual": ab["alpha_annualized"],
        "hit_ratio": hit_ratio(ret),
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) > 1 else 0.0,
    }


def rolling_window_metrics(
    equity: pd.Series,
    turnover: pd.Series,
    spy_close: pd.Series,
    window_days: int,
    prefix: str,
) -> list[dict]:
    rows = []
    for i in range(window_days, len(equity) + 1):
        sl = equity.iloc[i - window_days : i]
        to = turnover.iloc[i - window_days : i]
        label = f"{prefix}_{sl.index[-1].date()}"
        rows.append(_window_metrics(sl, to, spy_close, label))
    return rows


def run_subperiod_analysis(
    equity: pd.Series,
    turnover: pd.Series,
    spy_close: pd.Series,
) -> tuple[pd.DataFrame, dict]:
    eq = equity.dropna()
    mid = len(eq) // 2

    rows = [
        _window_metrics(eq.iloc[:mid], turnover.iloc[:mid], spy_close, "first_half"),
        _window_metrics(eq.iloc[mid:], turnover.iloc[mid:], spy_close, "second_half"),
    ]
    rows.extend(rolling_window_metrics(eq, turnover, spy_close, 126, "rolling_6m"))
    rows.extend(rolling_window_metrics(eq, turnover, spy_close, 252, "rolling_12m"))

    df = pd.DataFrame(rows)

    # Verdict logic
    sharpe_cols = df[df["period"].str.contains("rolling|half", regex=True)]["sharpe"]
    positive_frac = (sharpe_cols > 0).mean() if len(sharpe_cols) else 0.0

    first_ret = df.loc[df["period"] == "first_half", "total_return"]
    second_ret = df.loc[df["period"] == "second_half", "total_return"]
    total_pnl = first_ret.sum() + second_ret.sum()
    max_half_share = 0.0
    if total_pnl != 0 and len(first_ret) and len(second_ret):
        max_half_share = max(abs(first_ret.iloc[0]), abs(second_ret.iloc[0])) / abs(total_pnl)

    if positive_frac >= 0.55 and max_half_share < 0.85:
        verdict = "persistent"
    elif max_half_share >= 0.85:
        verdict = "single_regime_only"
    else:
        verdict = "unstable"

    summary = {
        "verdict": verdict,
        "pct_windows_sharpe_positive": float(positive_frac),
        "dominant_half_pnl_share": float(max_half_share),
        "first_half_sharpe": float(df.loc[df["period"] == "first_half", "sharpe"].iloc[0])
        if (df["period"] == "first_half").any()
        else np.nan,
        "second_half_sharpe": float(df.loc[df["period"] == "second_half", "sharpe"].iloc[0])
        if (df["period"] == "second_half").any()
        else np.nan,
    }
    return df, summary
