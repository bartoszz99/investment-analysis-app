"""
Multi-asset portfolio backtest.
Temporal: target_weights[t] decided with info <= t-1 -> execute at Open[t+1] via shift(1).
MTM at Close[t]. Fees: configurable bps; slippage scales with trailing vol.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from layers.backtest_engine import compute_metrics


def lag_weights_to_execution(weights: pd.DataFrame) -> pd.DataFrame:
    """exec_w[t] = target_w[t-1]; trade at Open[t]."""
    return weights.shift(1).fillna(0.0)


def _vol_scaled_slippage(base_bps: float, vol: float, coef: float) -> float:
    if np.isnan(vol) or vol <= 0:
        vol = 0.2
    return (base_bps / 10_000.0) * (1.0 + coef * vol)


def run_multi_asset_backtest(
    open_px: pd.DataFrame,
    close_px: pd.DataFrame,
    target_weights: pd.DataFrame,
    *,
    volume: pd.DataFrame | None = None,
    vol_panel: pd.DataFrame | None = None,
    start_capital: float = 100_000.0,
    fee_bps: float = 2.0,
    slippage_vol_coef: float = 0.10,
    tradable_mask: pd.Series | None = None,
) -> dict:
    """
    Daily rebalance: align exec weights at open, MTM at close.
    Returns portfolio metrics + per-asset equity contributions.
    """
    tickers = list(target_weights.columns)
    idx = open_px.index.intersection(close_px.index).intersection(target_weights.index)
    open_px = open_px.loc[idx, tickers]
    close_px = close_px.loc[idx, tickers]
    target_weights = target_weights.loc[idx, tickers]
    if vol_panel is not None:
        vol_panel = vol_panel.loc[idx, tickers]
    if volume is not None:
        volume = volume.loc[idx, tickers]

    if tradable_mask is None:
        tradable_mask = open_px.notna().all(axis=1) & close_px.notna().all(axis=1)
    else:
        tradable_mask = tradable_mask.loc[idx]

    exec_w = lag_weights_to_execution(target_weights)

    cash = float(start_capital)
    shares = pd.Series(0.0, index=tickers)
    equity_total: list[float] = []
    equity_by_asset = pd.DataFrame(0.0, index=idx, columns=tickers)
    turnover_daily: list[float] = []
    trades = 0
    prev_w = pd.Series(0.0, index=tickers)

    for dt in idx:
        if not tradable_mask.loc[dt]:
            equity_total.append(cash + float((shares * close_px.loc[dt]).sum()))
            equity_by_asset.loc[dt] = shares * close_px.loc[dt]
            turnover_daily.append(0.0)
            continue

        o = open_px.loc[dt]
        c = close_px.loc[dt]
        w = exec_w.loc[dt].fillna(0.0)

        nav_open = cash + float((shares * o).sum())
        if nav_open <= 0:
            nav_open = start_capital

        for t in tickers:
            target_val = nav_open * float(w[t])
            current_val = float(shares[t]) * float(o[t])
            delta_val = target_val - current_val
            if abs(delta_val) < 1e-6:
                continue

            vol = 0.2
            if vol_panel is not None and t in vol_panel.columns:
                vol = float(vol_panel.loc[dt, t]) if pd.notna(vol_panel.loc[dt, t]) else 0.2
            slip = _vol_scaled_slippage(fee_bps, vol, slippage_vol_coef)
            px = float(o[t])

            if delta_val > 0:
                buy_px = px * (1 + slip)
                fee = delta_val * (fee_bps / 10_000.0)
                invest = delta_val - fee
                if buy_px > 0:
                    shares[t] += invest / buy_px
                    cash -= delta_val
                    trades += 1
            else:
                sell_px = px * (1 - slip)
                sell_val = min(-delta_val, float(shares[t]) * px)
                if sell_px > 0 and sell_val > 0:
                    sh_sell = sell_val / px
                    sh_sell = min(sh_sell, shares[t])
                    proceeds = sh_sell * sell_px
                    fee = proceeds * (fee_bps / 10_000.0)
                    cash += proceeds - fee
                    shares[t] -= sh_sell
                    trades += 1

        nav_close = cash + float((shares * c).sum())
        equity_total.append(nav_close)
        equity_by_asset.loc[dt] = shares * c

        turnover_daily.append(float((w - prev_w).abs().sum() / 2.0))
        prev_w = w.copy()

    metrics = compute_metrics(equity_total, start_capital, trades)
    eq_series = pd.Series(equity_total, index=idx[: len(equity_total)], name="equity")

    peak = eq_series.cummax()
    drawdown = (eq_series - peak) / peak.replace(0, np.nan)

    return {
        **metrics,
        "equity_series": eq_series,
        "equity_by_asset": equity_by_asset,
        "drawdown": drawdown,
        "target_weights": target_weights,
        "exec_weights": exec_w,
        "turnover_daily": pd.Series(turnover_daily, index=idx[: len(turnover_daily)]),
    }
