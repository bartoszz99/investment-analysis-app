"""Causal regime detection — expanding thresholds, lagged output."""

import numpy as np
import pandas as pd

from layers.feature_engine import FeatureEngine

WINDOW = 20
MIN_HISTORY = 40

REGIME_TREND = "TREND"
REGIME_MEAN_REVERSION = "MEAN_REVERSION"
REGIME_HIGH_VOLATILITY = "HIGH_VOLATILITY"
ALL_REGIMES = (REGIME_TREND, REGIME_MEAN_REVERSION, REGIME_HIGH_VOLATILITY)


def _classify_row(v, s, vol_high, vol_low, slope_high, slope_low, slope_med) -> str:
    if v >= vol_high:
        return REGIME_HIGH_VOLATILITY
    if s >= slope_high and v < vol_high:
        return REGIME_TREND
    if s <= slope_low and v <= vol_low:
        return REGIME_MEAN_REVERSION
    if s >= slope_med:
        return REGIME_TREND
    return REGIME_MEAN_REVERSION


def detect_regime(df: pd.DataFrame, engine: FeatureEngine | None = None) -> pd.DataFrame:
    if "Close" not in df.columns:
        raise ValueError("Close required")

    engine = engine or FeatureEngine()
    close = df["Close"]
    features = pd.DataFrame(index=df.index)
    features["_vol_ratio"] = engine.vol_ratio(close, WINDOW)
    features["_slope_norm"] = engine.slope_normalized(close, WINDOW)

    raw_regime = pd.Series(index=df.index, dtype=object)
    for i in range(len(features)):
        v = features["_vol_ratio"].iloc[i]
        s = features["_slope_norm"].iloc[i]
        if pd.isna(v) or pd.isna(s):
            raw_regime.iloc[i] = np.nan
            continue
        hist = features.iloc[:i].dropna()
        if len(hist) < MIN_HISTORY:
            raw_regime.iloc[i] = np.nan
            continue
        vol = hist["_vol_ratio"]
        slope = hist["_slope_norm"]
        raw_regime.iloc[i] = _classify_row(
            v,
            s,
            vol.quantile(0.67),
            vol.quantile(0.33),
            slope.quantile(0.67),
            slope.quantile(0.33),
            slope.median(),
        )

    out = df.copy()
    # Extra lag: regime usable for decisions at t uses labels through t-1
    out["regime"] = raw_regime.shift(1)
    return out
