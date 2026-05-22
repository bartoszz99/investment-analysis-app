"""
Breadth hypothesis tests — A/B/C (pure research, no optimization).

A: ETF rises while breadth weakens → trend fragility
B: Breadth thrusts → continuation
C: High dispersion + weak breadth → reversal
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.common.forward_returns import HORIZONS, forward_return
from research.common.signal_evaluation import (
    evaluate_signal,
    long_short_decile_curve,
    quintile_spread,
    spearman_ic,
)


def _etf_momentum(etf_close: pd.Series, window: int = 20) -> pd.Series:
    lag = etf_close.shift(1)
    return lag / lag.shift(window) - 1.0


def hypothesis_a_fragility(
    breadth: pd.DataFrame,
    etf_close: pd.Series,
) -> dict:
    """
    Signal: rising ETF + falling breadth (divergence).
    Expect negative forward returns when fragility high.
    """
    mom = _etf_momentum(etf_close)
    breadth_chg = breadth["pct_above_sma20"].diff(5)
    fragility = (mom > 0).astype(float) * (-breadth_chg)
    thresh = fragility.expanding(min_periods=60).quantile(0.70)
    fragility = fragility.where(fragility > thresh)

    rows = []
    for h in HORIZONS:
        fwd = forward_return(etf_close, h)
        rows.append(
            evaluate_signal(
                fragility,
                fwd,
                signal_name="hypothesis_a_fragility",
                target=etf_close.name or "etf",
                horizon=h,
            )
        )
    return {
        "hypothesis": "A_trend_fragility",
        "mechanism": "Rally on narrowing participation → fragile trend",
        "evaluations": rows,
        "conditional_fwd_20d_rally_weak_breadth": _conditional_mean(
            fragility, forward_return(etf_close, 20), fragility.notna()
        ),
    }


def hypothesis_b_thrust(breadth: pd.DataFrame, etf_close: pd.Series) -> dict:
    """Breadth thrust → continuation."""
    thrust = breadth["breadth_thrust"]
    rows = []
    for h in HORIZONS:
        fwd = forward_return(etf_close, h)
        rows.append(
            evaluate_signal(
                thrust,
                fwd,
                signal_name="hypothesis_b_thrust",
                target=etf_close.name or "etf",
                horizon=h,
            )
        )
    return {
        "hypothesis": "B_breadth_thrust_continuation",
        "mechanism": "Broad participation surge → trend continuation",
        "evaluations": rows,
        "conditional_fwd_20d_top_thrust": _conditional_mean(
            thrust, forward_return(etf_close, 20), thrust > thrust.expanding(60).quantile(0.80)
        ),
    }


def hypothesis_c_reversal(breadth: pd.DataFrame, etf_close: pd.Series) -> dict:
    """High dispersion + weak breadth → reversal (expanding thresholds only)."""
    b = breadth["pct_above_sma20"]
    d = breadth["return_dispersion"]
    weak = b < b.expanding(min_periods=60).quantile(0.30)
    high_disp = d > d.expanding(min_periods=60).quantile(0.70)
    signal = (weak & high_disp).astype(float)
    signal[signal == 0] = np.nan

    rows = []
    for h in HORIZONS:
        fwd = forward_return(etf_close, h)
        rows.append(
            evaluate_signal(
                signal,
                fwd,
                signal_name="hypothesis_c_reversal",
                target=etf_close.name or "etf",
                horizon=h,
            )
        )
    return {
        "hypothesis": "C_dispersion_reversal",
        "mechanism": "Leadership concentration + weak breadth → mean reversion",
        "evaluations": rows,
        "conditional_fwd_5d": _conditional_mean(
            signal, forward_return(etf_close, 5), signal.notna()
        ),
    }


def _conditional_mean(
    signal: pd.Series,
    forward_ret: pd.Series,
    mask: pd.Series,
) -> float:
    aligned = pd.concat([signal, forward_ret], axis=1).dropna()
    m = mask.reindex(aligned.index).fillna(False)
    if m.sum() < 5:
        return float("nan")
    return float(aligned.loc[m, aligned.columns[-1]].mean())


def run_breadth_feature_ic(
    breadth: pd.DataFrame,
    etf_close: pd.Series,
    etf: str,
    neutral_features: dict[str, pd.Series] | None = None,
) -> list[dict]:
    """IC table for all breadth features vs ETF forward returns."""
    rows = []
    for col in breadth.columns:
        sig = breadth[col]
        for h in HORIZONS:
            fwd = forward_return(etf_close, h)
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


def decile_curves_for_features(
    breadth: pd.DataFrame,
    etf_close: pd.Series,
    features: list[str],
    horizon: int = 20,
) -> dict[str, pd.Series]:
    fwd = forward_return(etf_close, horizon)
    return {f: long_short_decile_curve(breadth[f], fwd) for f in features if f in breadth.columns}
