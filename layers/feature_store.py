"""
Central feature registry — lag before rolling, metadata, dependency graph.
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    category: str
    lookback: int
    lag: int = 1
    dependencies: tuple[str, ...] = ()
    leakage_safe: bool = True

    def __post_init__(self) -> None:
        if self.lag < 1:
            raise ValueError("lag must be >= 1 for causal features")


class FeatureStore:
    CATEGORIES = ("trend", "volatility", "mean_reversion", "cross_sectional", "microstructure_proxy")

    def __init__(self, use_cache: bool = True) -> None:
        self._registry: dict[str, FeatureSpec] = {}
        self._builders: dict[str, Callable[[pd.DataFrame], pd.Series]] = {}
        self._use_cache = use_cache
        self._cache = None
        if use_cache:
            from layers.feature_cache import FeatureCache

            self._cache = FeatureCache()
        self._register_defaults()

    def register(
        self,
        spec: FeatureSpec,
        builder: Callable[[pd.DataFrame], pd.Series],
    ) -> None:
        self._registry[spec.name] = spec
        self._builders[spec.name] = builder

    def get_spec(self, name: str) -> FeatureSpec:
        return self._registry[name]

    def list_features(self, category: str | None = None) -> list[str]:
        if category is None:
            return list(self._registry.keys())
        return [n for n, s in self._registry.items() if s.category == category]

    def dependency_graph(self) -> dict[str, list[str]]:
        return {name: list(spec.dependencies) for name, spec in self._registry.items()}

    def validate_no_future_access(self, series: pd.Series, raw: pd.Series) -> bool:
        """Heuristic: feature must not correlate perfectly with unlagged future return."""
        if len(series.dropna()) < 10:
            return True
        future = raw.pct_change().shift(-1)
        corr = series.corr(future)
        return not (corr is not None and abs(corr) > 0.99)

    def _lag(self, s: pd.Series, lag: int) -> pd.Series:
        return s.shift(lag)

    def _register_defaults(self) -> None:
        self.register(
            FeatureSpec("momentum_20d", "trend", 20, 1, ("Close",)),
            lambda df: self._momentum_20d(df),
        )
        self.register(
            FeatureSpec("realized_vol_20d", "volatility", 20, 1, ("Close",)),
            lambda df: self._realized_vol_20d(df),
        )
        self.register(
            FeatureSpec("zscore_10d", "mean_reversion", 10, 1, ("Close",)),
            lambda df: self._zscore_10d(df),
        )
        self.register(
            FeatureSpec("atr_14", "volatility", 14, 1, ("High", "Low", "Close")),
            lambda df: self._atr_14(df),
        )
        self.register(
            FeatureSpec("volume_spike", "microstructure_proxy", 20, 1, ("Volume",)),
            lambda df: self._volume_spike(df),
        )
        self.register(
            FeatureSpec("gap_return", "microstructure_proxy", 1, 1, ("Open", "Close")),
            lambda df: self._gap_return(df),
        )
        self.register(
            FeatureSpec("rolling_beta", "cross_sectional", 60, 1, ("Close",)),
            lambda df: self._rolling_beta(df),
        )

    def _momentum_20d(self, df: pd.DataFrame) -> pd.Series:
        c = self._lag(df["Close"], 1)
        return c / c.shift(20) - 1.0

    def _realized_vol_20d(self, df: pd.DataFrame) -> pd.Series:
        ret = self._lag(df["Close"], 1).pct_change()
        return ret.rolling(20, min_periods=20).std() * np.sqrt(252)

    def _zscore_10d(self, df: pd.DataFrame) -> pd.Series:
        c = self._lag(df["Close"], 1)
        m = c.rolling(10, min_periods=10).mean()
        s = c.rolling(10, min_periods=10).std()
        return (c - m) / s.replace(0, np.nan)

    def _atr_14(self, df: pd.DataFrame) -> pd.Series:
        h = self._lag(df["High"], 1)
        l = self._lag(df["Low"], 1)
        c = self._lag(df["Close"], 1)
        prev_c = c.shift(1)
        tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
        return tr.rolling(14, min_periods=14).mean()

    def _volume_spike(self, df: pd.DataFrame) -> pd.Series:
        v = self._lag(df["Volume"], 1)
        avg = v.rolling(20, min_periods=20).mean()
        return v / avg.replace(0, np.nan)

    def _gap_return(self, df: pd.DataFrame) -> pd.Series:
        o = self._lag(df["Open"], 1)
        prev_c = self._lag(df["Close"], 1).shift(1)
        return o / prev_c.replace(0, np.nan) - 1.0

    def _rolling_beta(self, df: pd.DataFrame, window: int = 60) -> pd.Series:
        ret = self._lag(df["Close"], 1).pct_change()
        mkt = ret.rolling(window, min_periods=window).mean()
        cov = ret.rolling(window, min_periods=window).cov(mkt)
        var = mkt.rolling(window, min_periods=window).var()
        return cov / var.replace(0, np.nan)

    def build(self, df: pd.DataFrame, names: list[str] | None = None) -> pd.DataFrame:
        names = names or self.list_features()
        if self._cache is not None:
            return self._cache.get_or_build(df, names, lambda d, n: self._build_uncached(d, n))

        return self._build_uncached(df, names)

    def _build_uncached(self, df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
        out = df.copy()
        for name in names:
            if name not in self._builders:
                raise KeyError(f"Unknown feature: {name}")
            out[name] = self._builders[name](df)
            spec = self._registry[name]
            if spec.lag > 1:
                out[name] = out[name].shift(spec.lag - 1)
        return out

    def build_matrix(self, df: pd.DataFrame, names: list[str] | None = None) -> pd.DataFrame:
        built = self.build(df, names)
        cols = names or self.list_features()
        return built[cols].copy()
