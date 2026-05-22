"""
Regime stability and tail-focused diagnostics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.equity_alpha_lab.hypothesis_tests import daily_ic, mean_ic


def spy_regime_masks(spy_close: pd.Series) -> dict[str, pd.Series]:
    lag = spy_close.shift(1)
    ma200 = lag.rolling(200, min_periods=200).mean()
    bull = lag > ma200
    vol = lag.pct_change().rolling(20, min_periods=20).std()
    high_vol = vol > vol.expanding(min_periods=60).median()
    return {
        "bull_market": bull,
        "bear_or_neutral": ~bull,
        "high_vol": high_vol,
        "low_vol": ~high_vol,
    }


def regime_ic_table(
    signal: pd.DataFrame,
    fwd: pd.DataFrame,
    spy_close: pd.Series,
) -> list[dict]:
    masks = spy_regime_masks(spy_close)
    rows = []
    for regime, mask in masks.items():
        sub_idx = mask.reindex(signal.index).fillna(False)
        ic_series = daily_ic(signal, fwd)
        on = ic_series[sub_idx.reindex(ic_series.index, fill_value=False)]
        rows.append(
            {
                "feature": getattr(signal, "name", "panel"),
                "regime": regime,
                "mean_ic": float(on.mean()) if len(on) else np.nan,
                "n_days": int(len(on)),
            }
        )
    return rows


def build_regime_report(
    features: dict[str, pd.DataFrame],
    fwd_5d: pd.DataFrame,
    spy_close: pd.Series,
) -> pd.DataFrame:
    rows = []
    for name, sig in features.items():
        for r in regime_ic_table(sig, fwd_5d, spy_close):
            r["feature"] = name
            rows.append(r)
    return pd.DataFrame(rows)
