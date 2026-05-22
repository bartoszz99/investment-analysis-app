"""
Per-ticker feature engine for multi-asset panels.
Anti-leakage: shift(1) before all rolling stats; no cross-asset inputs in feature calc.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from layers.data_layer import MultiAssetOHLCV
from layers.feature_engine import FeatureEngine


class MultiAssetFeatureEngine:
    """
    Builds causal features independently for each ticker, then optional wide matrices.
    feature[t] uses OHLCV <= t-1 only.
    """

    SMA_SHORT = 10
    SMA_LONG = 30
    VOL_WINDOW = 20
    MOM_WINDOW = 20

    def __init__(self) -> None:
        self._engine = FeatureEngine()

    def build_ticker_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Single-ticker feature frame (causal)."""
        out = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        close = out["Close"]

        out["SMA_10"] = self._engine.sma(close, self.SMA_SHORT)
        out["SMA_30"] = self._engine.sma(close, self.SMA_LONG)

        lagged = close.shift(1)
        out["ret_1d"] = lagged.pct_change(1)
        out["ret_5d"] = lagged.pct_change(5)
        out["volatility_20d"] = lagged.pct_change().rolling(
            self.VOL_WINDOW, min_periods=self.VOL_WINDOW
        ).std() * np.sqrt(252)
        out["momentum_20d"] = lagged / lagged.shift(self.MOM_WINDOW) - 1.0

        return out

    def build_panel(self, bundle: MultiAssetOHLCV) -> dict[str, pd.DataFrame]:
        """Per-ticker features — no information flows between tickers."""
        tradable = bundle.drop_non_tradable()
        return {t: self.build_ticker_features(tradable.panels[t]) for t in tradable.tickers}

    def to_wide(
        self,
        feature_panels: dict[str, pd.DataFrame],
        feature_name: str,
    ) -> pd.DataFrame:
        """Wide feature matrix for cross-sectional ops (index=date, columns=ticker)."""
        return pd.DataFrame(
            {t: feature_panels[t][feature_name] for t in feature_panels},
        ).sort_index()
