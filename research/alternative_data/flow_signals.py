"""
Flow pressure signals — volume/dollar-volume proxies from OHLCV.
No external data; causal lags on price/volume only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.alternative_data.base import (
    AlternativeSignal,
    LeakageRisk,
    SignalMetadata,
    apply_lag,
    expanding_zscore,
)


def abnormal_volume_factor(volume: pd.Series, window: int = 20, lag_days: int = 1) -> pd.Series:
    """Volume / rolling mean volume (lagged)."""
    v = apply_lag(volume, lag_days)
    avg = v.rolling(window, min_periods=window).mean()
    return v / avg.replace(0, np.nan)


def dollar_volume_zscore(
    close: pd.Series,
    volume: pd.Series,
    window: int = 20,
    lag_days: int = 1,
) -> pd.Series:
    dv = apply_lag(close, lag_days) * apply_lag(volume, lag_days)
    mu = dv.rolling(window, min_periods=window).mean()
    sigma = dv.rolling(window, min_periods=window).std()
    return (dv - mu) / sigma.replace(0, np.nan)


def breadth_thrust(close_wide: pd.DataFrame, lag_days: int = 1) -> pd.Series:
    """Fraction of universe with positive 1d return (lagged)."""
    ret = close_wide.pct_change()
    ret_lag = apply_lag(ret, lag_days)
    return (ret_lag > 0).mean(axis=1)


def build_flow_signals(
    close_wide: pd.DataFrame,
    volume_wide: pd.DataFrame,
    lag_days: int = 1,
) -> dict[str, AlternativeSignal]:
    """
    Per-ETF flow proxies + universe flow_pressure_factor (mean abnormal vol z).
    """
    meta = {
        "source": "OHLCV volume (same framework as ETF prices)",
        "lag_days": lag_days,
        "update_frequency": "daily",
        "leakage_risk": LeakageRisk.LOW,
        "timestamp_assumption": "Volume known after session close; usable next open",
        "lag_policy": f"shift({lag_days}) on volume/close before rolling stats",
    }

    signals: dict[str, AlternativeSignal] = {}
    abn_list = []
    dvz_list = []

    for ticker in close_wide.columns:
        abn = abnormal_volume_factor(volume_wide[ticker], lag_days=lag_days)
        dvz = dollar_volume_zscore(close_wide[ticker], volume_wide[ticker], lag_days=lag_days)
        abn_list.append(abn)
        dvz_list.append(dvz)
        signals[f"flow_abnormal_vol_{ticker}"] = AlternativeSignal(
            metadata=SignalMetadata(
                name=f"flow_abnormal_vol_{ticker}",
                description=f"Abnormal volume {ticker}",
                **meta,
            ),
            series=abn,
        )
        signals[f"flow_dv_zscore_{ticker}"] = AlternativeSignal(
            metadata=SignalMetadata(
                name=f"flow_dv_zscore_{ticker}",
                description=f"Dollar volume z-score {ticker}",
                **meta,
            ),
            series=dvz,
        )

    thrust = breadth_thrust(close_wide, lag_days=lag_days)
    signals["flow_breadth_thrust"] = AlternativeSignal(
        metadata=SignalMetadata(
            name="flow_breadth_thrust",
            description="Universe breadth (% positive returns)",
            **meta,
        ),
        series=thrust,
    )

    pressure = pd.concat(abn_list, axis=1).mean(axis=1)
    pressure = expanding_zscore(pressure)
    signals["flow_pressure_factor"] = AlternativeSignal(
        metadata=SignalMetadata(
            name="flow_pressure_factor",
            description="Mean abnormal volume z-score across universe",
            **meta,
        ),
        series=pressure,
    )

    return signals
