"""
Cross-sectional factor framework (single-asset time series as degenerate panel).
Z-score ONLY on expanding/rolling past windows — no global full-sample stats.
"""

import numpy as np
import pandas as pd

from layers.feature_store import FeatureStore


def _rolling_zscore(s: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Z-score using only past `window` observations (causal)."""
    min_periods = min_periods or window
    lagged = s.shift(1)
    m = lagged.rolling(window, min_periods=min_periods).mean()
    sd = lagged.rolling(window, min_periods=min_periods).std()
    return (lagged - m) / sd.replace(0, np.nan)


def _gram_schmidt(columns: pd.DataFrame) -> pd.DataFrame:
    """Orthogonalize factor columns row-wise where possible."""
    out = columns.copy().astype(float)
    cols = list(out.columns)
    for i, col in enumerate(cols):
        v = out[col].to_numpy(dtype=float)
        for prev in cols[:i]:
            u = out[prev].to_numpy(dtype=float)
            denom = np.dot(u, u)
            if denom > 1e-12:
                v = v - np.dot(v, u) / denom * u
        out[col] = v
    return out


class FactorModel:
    FACTORS = ("momentum", "volatility", "mean_reversion", "trend_slope", "volume_shock")

    def __init__(self, zscore_window: int = 60) -> None:
        self.zscore_window = zscore_window
        self.store = FeatureStore()

    def raw_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        built = self.store.build(df)
        close = df["Close"]
        lag_c = close.shift(1)
        out = pd.DataFrame(index=df.index)
        out["momentum"] = built.get("momentum_20d", lag_c / lag_c.shift(20) - 1)
        out["volatility"] = built.get("realized_vol_20d", lag_c.pct_change().rolling(20).std())
        out["mean_reversion"] = -built.get("zscore_10d", _rolling_zscore(lag_c, 10))
        slope = built.get("momentum_20d", lag_c.pct_change(20))
        out["trend_slope"] = slope
        out["volume_shock"] = built.get("volume_spike", df["Volume"].shift(1))
        return out

    def orthogonal_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        raw = self.raw_factors(df)
        zd = pd.DataFrame(index=df.index)
        for col in self.FACTORS:
            if col in raw.columns:
                zd[col] = _rolling_zscore(raw[col], self.zscore_window)
        return _gram_schmidt(zd)

    def factor_exposures(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.orthogonal_factors(df)

    def information_coefficient(self, factors: pd.DataFrame, forward_ret: pd.Series) -> pd.Series:
        """IC per factor — forward_ret is label (use only in historical analysis)."""
        ic = {}
        fwd = forward_ret.shift(-1)
        for col in factors.columns:
            ic[col] = factors[col].corr(fwd)
        return pd.Series(ic)

    def summarize(self, df: pd.DataFrame) -> dict:
        fac = self.factor_exposures(df)
        fwd = df["Close"].pct_change().shift(-5)
        ic = self.information_coefficient(fac, fwd)
        return {
            "factor_exposures_tail": fac.tail(3).to_dict(),
            "ic": ic.to_dict(),
            "zscore_window": self.zscore_window,
        }
