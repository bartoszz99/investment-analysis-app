"""
Parameter sensitivity — diagnostic counterfactual grid only.
Same execution engine (Open[t+1], fees, slippage); alternate signal params only.
Does NOT change production config or select best params for live use.
"""

from __future__ import annotations

import pandas as pd

from analysis.robustness_common import (
    annualized_sharpe,
    cagr,
    excess_return_vs_spy,
    fragility_score,
    max_drawdown_pct,
    ols_alpha_beta,
)
from layers.cross_sectional import apply_max_weight_cap, top_k_equal_weight_weights
from layers.multi_asset_backtest import run_multi_asset_backtest

MOMENTUM_WINDOWS = [5, 10, 20, 40, 60]
TOP_K_VALUES = [1, 2, 3, 4]
REBALANCE_FREQ = {"1D": 1, "5D": 5, "10D": 10}


def build_momentum_wide(
    feature_panels: dict[str, pd.DataFrame],
    window: int,
) -> pd.DataFrame:
    """Causal momentum: lag(1) close / lag(1+window) close - 1."""
    cols = {}
    for ticker, df in feature_panels.items():
        lagged = df["Close"].shift(1)
        cols[ticker] = lagged / lagged.shift(window) - 1.0
    return pd.DataFrame(cols).sort_index()


def build_top_k_weights(
    momentum_wide: pd.DataFrame,
    top_k: int,
    max_weight: float = 0.4,
) -> pd.DataFrame:
    w = top_k_equal_weight_weights(momentum_wide, k=top_k, max_weight=max_weight)
    return apply_max_weight_cap(w, max_weight)


def apply_rebalance_frequency(weights: pd.DataFrame, freq_days: int) -> pd.DataFrame:
    """Hold weights between rebalance dates (diagnostic only)."""
    if freq_days <= 1:
        return weights
    w = weights.copy()
    rebalance = pd.Series(False, index=w.index)
    rebalance.iloc[0] = True
    for i in range(len(w)):
        if i % freq_days == 0:
            rebalance.iloc[i] = True
    return w.where(rebalance).ffill().fillna(0.0)


def run_parameter_grid(
    *,
    feature_panels: dict[str, pd.DataFrame],
    open_wide: pd.DataFrame,
    close_wide: pd.DataFrame,
    vol_wide: pd.DataFrame,
    vol_feat: pd.DataFrame,
    tradable_mask: pd.Series,
    start_capital: float,
    fee_bps: float,
    slippage_vol_coef: float,
    max_weight: float,
    momentum_windows: list[int] | None = None,
    top_k_values: list[int] | None = None,
    rebalance_freqs: dict[str, int] | None = None,
) -> pd.DataFrame:
    momentum_windows = momentum_windows or MOMENTUM_WINDOWS
    top_k_values = top_k_values or TOP_K_VALUES
    rebalance_freqs = rebalance_freqs or REBALANCE_FREQ

    spy_close = close_wide["SPY"]
    rows = []

    for mw in momentum_windows:
        mom = build_momentum_wide(feature_panels, mw)
        for k in top_k_values:
            daily_w = build_top_k_weights(mom, k, max_weight)
            for freq_label, freq_days in rebalance_freqs.items():
                weights = apply_rebalance_frequency(daily_w, freq_days)
                bt = run_multi_asset_backtest(
                    open_wide,
                    close_wide,
                    weights,
                    volume=vol_wide,
                    vol_panel=vol_feat,
                    start_capital=start_capital,
                    fee_bps=fee_bps,
                    slippage_vol_coef=slippage_vol_coef,
                    tradable_mask=tradable_mask,
                )
                eq = bt["equity_series"]
                ret = eq.pct_change().dropna()
                spy_ret = spy_close.pct_change().reindex(ret.index)
                ab = ols_alpha_beta(ret, spy_ret)
                rows.append(
                    {
                        "momentum_window": mw,
                        "top_k": k,
                        "rebalance_freq": freq_label,
                        "cagr": cagr(eq),
                        "sharpe": annualized_sharpe(ret),
                        "max_drawdown_pct": max_drawdown_pct(eq),
                        "turnover_annual": float(bt["turnover_daily"].mean() * 252),
                        "excess_return_vs_spy": excess_return_vs_spy(eq, spy_close),
                        "beta_vs_spy": ab["beta"],
                        "alpha_annualized": ab["alpha_annualized"],
                        "return_pct": bt["return_pct"],
                        "trades": bt["trades"],
                    }
                )

    surface = pd.DataFrame(rows)
    surface["fragility_score_global"] = fragility_score(surface["sharpe"].tolist())
    return surface
