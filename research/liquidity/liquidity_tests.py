"""
Liquidity exhaustion hypothesis tests — D/E.

D: Extreme extension + abnormal volume → mean reversion
E: High volume + weak follow-through → exhaustion
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.common.forward_returns import HORIZONS, forward_return
from research.common.signal_evaluation import evaluate_signal


def hypothesis_d_mean_reversion(features: pd.DataFrame, close: pd.Series, etf: str) -> dict:
    """
    Signal: high |3d return| + high volume z-score → expect negative fwd return (reversal).
    Use signed extension * volume z as exhaustion intensity; IC vs forward return.
    """
    ext = features["return_3d_extension"]
    vol_z = features["volume_zscore_20d"]
    signal = ext.abs() * vol_z
    ext_thresh = ext.abs().expanding(min_periods=60).quantile(0.70)
    signal = signal.where((ext.abs() > ext_thresh) & (vol_z > 1.0))

    rows = []
    for h in HORIZONS:
        fwd = forward_return(close, h)
        # Reversal: high exhaustion → negative forward return; flip sign for IC interpretability
        rows.append(
            evaluate_signal(
                -signal,
                fwd,
                signal_name="hypothesis_d_exhaustion_reversal",
                target=etf,
                horizon=h,
            )
        )
    return {
        "hypothesis": "D_extension_volume_reversal",
        "mechanism": "Forced chase exhausts → short-term reversal",
        "other_side": "Late momentum chasers / short covering completion",
        "evaluations": rows,
    }


def hypothesis_e_weak_followthrough(features: pd.DataFrame, close: pd.Series, etf: str) -> dict:
    """
    High volume day with small next-bar continuation proxy:
    volume spike today, weak 1d return magnitude vs volume signal.
    """
    vol_z = features["volume_zscore_20d"]
    ret1 = close.shift(1).pct_change(1)
    follow_through = ret1.abs()
    weak = follow_through < follow_through.rolling(20, min_periods=20).median()
    high_vol = vol_z > vol_z.expanding(min_periods=60).quantile(0.80)
    signal = (high_vol & weak).astype(float)
    signal[signal == 0] = np.nan

    rows = []
    for h in HORIZONS:
        fwd = forward_return(close, h)
        rows.append(
            evaluate_signal(
                -signal,
                fwd,
                signal_name="hypothesis_e_volume_no_followthrough",
                target=etf,
                horizon=h,
            )
        )
    return {
        "hypothesis": "E_volume_exhaustion",
        "mechanism": "High volume without follow-through → trapped participants",
        "other_side": "Exhausted buyers/sellers at short-term extremes",
        "evaluations": rows,
    }


def run_liquidity_feature_ic(
    features: pd.DataFrame,
    close: pd.Series,
    etf: str,
    neutral_features: dict[str, pd.Series] | None = None,
) -> list[dict]:
    rows = []
    for col in features.columns:
        sig = features[col]
        for h in HORIZONS:
            fwd = forward_return(close, h)
            neutral = neutral_features.get(col) if neutral_features else None
            rows.append(
                evaluate_signal(
                    sig,
                    fwd,
                    signal_name=col,
                    target=etf,
                    horizon=h,
                    neutral_signal=neutral,
                )
            )
    return rows
