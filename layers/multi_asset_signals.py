"""
Multi-asset signal generation — per-ticker scores + portfolio weights.
Temporal: weights[t] use features at t (already lagged per ticker) -> execute Open[t+1].
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from layers.cross_sectional import (
    apply_max_weight_cap,
    cross_sectional_rank,
    cross_sectional_zscore,
    top_k_equal_weight_weights,
)
from layers.multi_asset_features import MultiAssetFeatureEngine
from layers.portfolio_optimizer import risk_parity_weights, rolling_covariance


def momentum_top_k_weights(
    momentum_wide: pd.DataFrame,
    k: int = 2,
    *,
    short_enabled: bool = False,
    max_weight: float = 0.4,
) -> pd.DataFrame:
    """Rank by momentum; allocate equal weight to top k (long-only default)."""
    w = top_k_equal_weight_weights(
        momentum_wide,
        k=k,
        short_enabled=short_enabled,
        max_weight=max_weight,
    )
    return apply_max_weight_cap(w, max_weight)


def risk_parity_target_weights(
    returns_wide: pd.DataFrame,
    tradable_mask: pd.Series,
    *,
    cov_window: int = 60,
    max_weight: float = 0.4,
) -> pd.DataFrame:
    """
  Daily risk-parity weights using trailing covariance only (causal).
  Recomputed each day from returns <= t-1.
    """
    tickers = list(returns_wide.columns)
    weights = pd.DataFrame(0.0, index=returns_wide.index, columns=tickers)

    for i, dt in enumerate(returns_wide.index):
        if not tradable_mask.loc[dt]:
            continue
        hist = returns_wide.iloc[:i].tail(cov_window).dropna(how="all")
        if len(hist) < 5:
            n = len(tickers)
            weights.loc[dt] = 1.0 / n
            continue
        cov = rolling_covariance(hist, min(cov_window, len(hist)))
        w = risk_parity_weights(cov)
        w = np.clip(w, 0, max_weight)
        w = w / w.sum() if w.sum() > 0 else np.ones(len(tickers)) / len(tickers)
        weights.loc[dt] = w

    return weights


def build_momentum_portfolio_weights(
    feature_panels: dict[str, pd.DataFrame],
    *,
    top_k: int = 2,
    short_enabled: bool = False,
    max_weight: float = 0.4,
    method: str = "equal_weight",
    tradable_mask: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (raw_weights, zscore_momentum, ranks).
    method: equal_weight top-k | risk_parity on universe returns.
    """
    engine = MultiAssetFeatureEngine()
    mom = engine.to_wide(feature_panels, "momentum_20d")
    z = cross_sectional_zscore(mom)
    ranks = cross_sectional_rank(mom, ascending=False)

    if method == "risk_parity":
        ret = engine.to_wide(feature_panels, "ret_1d")
        mask = tradable_mask if tradable_mask is not None else mom.notna().all(axis=1)
        w = risk_parity_target_weights(ret, mask, max_weight=max_weight)
    else:
        w = momentum_top_k_weights(mom, k=top_k, short_enabled=short_enabled, max_weight=max_weight)

    return w, z, ranks


def signal_matrix_from_weights(weights: pd.DataFrame) -> pd.DataFrame:
    """Map portfolio weights to per-ticker signal in [-1, 1] (long-only: [0, 1])."""
    return weights.clip(-1.0, 1.0)
