"""
Forward return distribution analysis after structural events.
Focus on tails, not just means.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.common.forward_returns import forward_return

HORIZONS = (5, 10, 20)


def drawdown_over_horizon(close: pd.Series, horizon: int) -> pd.Series:
    """Max drawdown from t+1 through t+horizon (close-based)."""
    out = pd.Series(np.nan, index=close.index)
    for i in range(len(close) - horizon):
        path = close.iloc[i + 1 : i + horizon + 1]
        if path.isna().any():
            continue
        peak = path.cummax()
        dd = (path - peak) / peak.replace(0, np.nan)
        out.iloc[i] = dd.min()
    return out


def distribution_stats(forward_rets: pd.Series, drawdowns: pd.Series | None = None) -> dict:
    r = forward_rets.dropna()
    if len(r) < 5:
        return {"n": len(r)}

    stats = {
        "n": int(len(r)),
        "mean": float(r.mean()),
        "median": float(r.median()),
        "std": float(r.std()),
        "volatility_ann": float(r.std() * np.sqrt(252 / max(len(r) / len(r) * 20, 1))),
        "p05": float(r.quantile(0.05)),
        "p25": float(r.quantile(0.25)),
        "p75": float(r.quantile(0.75)),
        "p95": float(r.quantile(0.95)),
        "skewness": float(r.skew()),
        "kurtosis": float(r.kurtosis()),
        "left_tail_freq": float((r < r.quantile(0.10)).mean()),
        "prob_correction_gt_5pct": float((r < -0.05).mean()),
        "prob_continued_rally_gt_2pct": float((r > 0.02).mean()),
    }
    if drawdowns is not None:
        dd = drawdowns.reindex(r.index).dropna()
        if len(dd) >= 3:
            stats["mean_max_drawdown"] = float(dd.mean())
            stats["drawdown_prob_gt_3pct"] = float((dd < -0.03).mean())
    return stats


def analyze_event_distribution(
    event_mask: pd.Series,
    etf_close: pd.Series,
    horizons: tuple[int, ...] = HORIZONS,
) -> list[dict]:
    """Full distribution metrics for each horizon after event days."""
    rows = []
    for h in horizons:
        fwd = forward_return(etf_close, h)
        dd = drawdown_over_horizon(etf_close, h)
        event_days = event_mask.fillna(False)
        stats = distribution_stats(fwd[event_days], dd[event_days])
        stats["horizon"] = h
        stats["event_count"] = int(event_days.sum())
        rows.append(stats)
    return rows


def compare_to_unconditional(
    event_mask: pd.Series,
    etf_close: pd.Series,
    horizon: int = 20,
) -> dict:
    """Event vs all-days distribution comparison."""
    fwd = forward_return(etf_close, horizon)
    event_stats = distribution_stats(fwd[event_mask.fillna(False)])
    base_stats = distribution_stats(fwd)
    return {
        "horizon": horizon,
        "event": event_stats,
        "unconditional": base_stats,
        "mean_diff": event_stats.get("mean", np.nan) - base_stats.get("mean", np.nan),
        "left_tail_diff": event_stats.get("left_tail_freq", np.nan)
        - base_stats.get("left_tail_freq", np.nan),
        "correction_prob_diff": event_stats.get("prob_correction_gt_5pct", np.nan)
        - base_stats.get("prob_correction_gt_5pct", np.nan),
    }
