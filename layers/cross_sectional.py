"""
Cross-sectional layer — ranks and z-scores across assets on each date.
Uses only contemporaneous feature values already computed with per-asset lag;
no future dates and no cross-asset feature construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cross_sectional_zscore(wide: pd.DataFrame) -> pd.DataFrame:
    """
    z_i,t = (x_i,t - mean_t) / std_t across assets at date t.
    Rows with <2 valid assets -> NaN.
    """
    def _z(row: pd.Series) -> pd.Series:
        valid = row.dropna()
        if len(valid) < 2:
            return pd.Series(np.nan, index=row.index)
        mu = valid.mean()
        sigma = valid.std()
        if sigma == 0 or np.isnan(sigma):
            return pd.Series(0.0, index=row.index)
        return (row - mu) / sigma

    return wide.apply(_z, axis=1)


def cross_sectional_rank(wide: pd.DataFrame, ascending: bool = False) -> pd.DataFrame:
    """Rank ETFs per day (1 = best if ascending=False for momentum)."""
    return wide.rank(axis=1, ascending=ascending, method="average")


def top_k_equal_weight_weights(
    score_wide: pd.DataFrame,
    k: int = 2,
    *,
    short_enabled: bool = False,
    short_k: int = 1,
    max_weight: float = 0.4,
) -> pd.DataFrame:
    """
    Long-only top-k equal weight by cross-sectional score (higher = better).
    If short_enabled: long top k, short bottom short_k with -1/(k+short_k) style;
    default long-only sums to 1.
    """
    tickers = list(score_wide.columns)
    weights = pd.DataFrame(0.0, index=score_wide.index, columns=tickers)

    for dt, row in score_wide.iterrows():
        valid = row.dropna()
        if len(valid) < max(k, 1):
            continue
        ranked = valid.rank(ascending=False, method="first")
        longs = ranked[ranked <= k].index.tolist()
        if not longs:
            continue

        if short_enabled and len(valid) >= k + short_k:
            shorts = ranked[ranked > len(valid) - short_k].index.tolist()
            n_side = len(longs) + len(shorts)
            w_long = 1.0 / n_side
            w_short = -1.0 / n_side
            for t in longs:
                weights.loc[dt, t] = w_long
            for t in shorts:
                weights.loc[dt, t] = w_short
        else:
            w = min(1.0 / len(longs), max_weight)
            for t in longs:
                weights.loc[dt, t] = w
            s = weights.loc[dt].sum()
            if s > 0 and abs(s - 1.0) > 1e-9:
                weights.loc[dt] /= s

    return weights


def apply_max_weight_cap(weights: pd.DataFrame, max_weight: float = 0.4) -> pd.DataFrame:
    """Clip and renormalize long-only weights per row."""
    out = weights.clip(lower=0.0)
    for dt in out.index:
        row = out.loc[dt]
        if row.sum() <= 0:
            continue
        capped = row.clip(upper=max_weight)
        s = capped.sum()
        out.loc[dt] = capped / s if s > 0 else capped
    return out
