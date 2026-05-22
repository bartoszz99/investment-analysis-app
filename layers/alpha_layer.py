"""
Ensemble alpha layer — probabilistic forecasts in [-1, 1].
"""

import math

import numpy as np
import pandas as pd

from layers.feature_store import FeatureStore


class AlphaModel:
    def __init__(self, feature_store: FeatureStore | None = None) -> None:
        self.store = feature_store or FeatureStore()
        self.weights: dict[str, float] = {
            "alpha_trend": 0.35,
            "alpha_mean_reversion": 0.25,
            "alpha_volatility": 0.20,
            "alpha_cross_sectional": 0.20,
        }

    def set_weights(self, weights: dict[str, float]) -> None:
        total = sum(weights.values())
        self.weights = {k: v / total for k, v in weights.items()} if total > 0 else weights

    def _trend_signal(self, df: pd.DataFrame) -> pd.Series:
        mom = self.store.build(df, ["momentum_20d"])["momentum_20d"]
        return np.tanh(mom.fillna(0) * 5)

    def _mean_reversion_signal(self, df: pd.DataFrame) -> pd.Series:
        z = self.store.build(df, ["zscore_10d"])["zscore_10d"]
        return -np.tanh(z.fillna(0))

    def _volatility_signal(self, df: pd.DataFrame) -> pd.Series:
        vol = self.store.build(df, ["realized_vol_20d"])["realized_vol_20d"]
        med = vol.expanding(min_periods=20).median()
        rel = (vol / med.replace(0, np.nan) - 1.0).fillna(0)
        return -np.tanh(rel)

    def _cross_sectional_signal(self, df: pd.DataFrame) -> pd.Series:
        beta = self.store.build(df, ["rolling_beta"])["rolling_beta"]
        return np.tanh((beta.fillna(1) - 1) * 2)

    def component_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["alpha_trend"] = self._trend_signal(df)
        out["alpha_mean_reversion"] = self._mean_reversion_signal(df)
        out["alpha_volatility"] = self._volatility_signal(df)
        out["alpha_cross_sectional"] = self._cross_sectional_signal(df)
        return out

    def forecast(self, df: pd.DataFrame, dynamic_weights: bool = False) -> pd.DataFrame:
        comp = self.component_signals(df)
        w = dict(self.weights)
        if dynamic_weights:
            vol = self.store.build(df, ["realized_vol_20d"])["realized_vol_20d"]
            inv_vol = 1.0 / vol.replace(0, np.nan)
            scale = inv_vol / inv_vol.expanding(min_periods=20).mean()
            for key in w:
                w[key] = w[key] * scale.fillna(1).iloc[-1] if len(scale) else w[key]
            s = sum(w.values())
            w = {k: v / s for k, v in w.items()}

        raw = sum(comp[k] * w[k] for k in w if k in comp.columns)
        result = df.copy()
        result["forecast"] = np.tanh(raw)
        result["signal_weight"] = (result["forecast"] + 1) / 2
        result["position"] = (result["forecast"] > 0).astype(int)
        return result
