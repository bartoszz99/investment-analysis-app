"""
Causal feature engineering.
Rule: feature[t] = f(data <= t-1) via input shift(1) before rolling stats.
"""

import numpy as np
import pandas as pd

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    lookback: int
    causal: bool = True
    note: str = "Uses shift(1) before rolling; no current bar in window"


class FeatureEngine:
    """Leakage-safe feature transforms (SMA / rolling stats)."""

    @staticmethod
    def _lag(series: pd.Series) -> pd.Series:
        """Data known at start of bar t = through t-1."""
        return series.shift(1)

    def sma(self, close: pd.Series, window: int) -> pd.Series:
        lagged = self._lag(close)
        return lagged.rolling(window=window, min_periods=window).mean()

    def add_sma_features(
        self, df: pd.DataFrame, short: int, long: int, column: str = "Close"
    ) -> pd.DataFrame:
        out = df.copy()
        out[f"SMA_{short}"] = self.sma(out[column], short)
        out[f"SMA_{long}"] = self.sma(out[column], long)
        return out

    def rolling_mean(self, close: pd.Series, window: int) -> pd.Series:
        return self._lag(close).rolling(window=window, min_periods=window).mean()

    def rolling_std(self, close: pd.Series, window: int) -> pd.Series:
        return self._lag(close).rolling(window=window, min_periods=window).std()

    def rolling_slope(self, close: pd.Series, window: int) -> pd.Series:
        lagged = self._lag(close)
        x = np.arange(window, dtype=float)
        x_mean = x.mean()
        x_var = ((x - x_mean) ** 2).sum()
        values = lagged.to_numpy(dtype=float)
        slopes = []
        for i in range(len(values)):
            if i < window - 1:
                slopes.append(np.nan)
                continue
            y = values[i - window + 1 : i + 1]
            if np.any(np.isnan(y)):
                slopes.append(np.nan)
                continue
            y_mean = y.mean()
            cov = np.sum((x - x_mean) * (y - y_mean))
            slopes.append(cov / x_var if x_var else np.nan)
        return pd.Series(slopes, index=close.index)

    def vol_ratio(self, close: pd.Series, window: int) -> pd.Series:
        mean = self.rolling_mean(close, window)
        std = self.rolling_std(close, window)
        return std / mean.replace(0, np.nan)

    def slope_normalized(self, close: pd.Series, window: int) -> pd.Series:
        mean = self.rolling_mean(close, window)
        slope = self.rolling_slope(close, window)
        return (slope / mean.replace(0, np.nan)).abs()

    def dependency_graph(self) -> dict[str, list[str]]:
        from layers.feature_store import FeatureStore

        base = {
            "Close": [],
            "SMA_*": ["Close (lagged)"],
            "roll_mean": ["Close (lagged)"],
            "roll_std": ["Close (lagged)"],
        }
        base.update(FeatureStore().dependency_graph())
        return base
