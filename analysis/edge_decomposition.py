"""
Edge decomposition — diagnostic layer only.
Does NOT alter production signals, portfolio construction, or execution.

Decomposes: market beta vs alpha, per-ETF PnL, momentum vs rotation counterfactuals,
rolling edge stability (3m / 6m / 12m Sharpe).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from layers.cross_sectional import apply_max_weight_cap, cross_sectional_zscore, top_k_equal_weight_weights
from layers.multi_asset_backtest import run_multi_asset_backtest


def _annualized_sharpe(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) < 2 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(252))


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity - peak) / peak.replace(0, np.nan)
    return float(abs(dd.min()) * 100) if len(dd) else 0.0


def ols_portfolio_vs_spy(
    portfolio_returns: pd.Series,
    spy_returns: pd.Series,
) -> dict:
    """
    R_portfolio = alpha + beta * R_SPY + epsilon
    Returns daily alpha (intercept), beta, R², annualized alpha.
    """
    aligned = pd.concat(
        [portfolio_returns.rename("p"), spy_returns.rename("s")],
        axis=1,
    ).dropna()
    if len(aligned) < 10:
        return {
            "alpha_daily": np.nan,
            "alpha_annualized": np.nan,
            "beta": np.nan,
            "r_squared": np.nan,
            "n_obs": len(aligned),
        }

    y = aligned["p"].to_numpy()
    x = aligned["s"].to_numpy()
    X = np.column_stack([np.ones(len(x)), x])
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = float(coeffs[0]), float(coeffs[1])
    y_hat = X @ coeffs
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "alpha_daily": alpha,
        "alpha_annualized": alpha * 252,
        "beta": beta,
        "r_squared": r2,
        "n_obs": len(aligned),
        "residual_vol_annual": float(np.std(y - y_hat) * np.sqrt(252)),
    }


def pnl_attribution_by_asset(
    equity_by_asset: pd.DataFrame,
    portfolio_equity: pd.Series,
    exec_weights: pd.DataFrame,
    close_wide: pd.DataFrame,
) -> pd.DataFrame:
    """
    Per-ETF contribution to return and volatility (diagnostic).
    Return contrib: sum of daily PnL share; vol contrib: weight * vol * corr to portfolio.
    """
    asset_pnl = equity_by_asset.diff()
    port_pnl = portfolio_equity.diff()
    port_ret = portfolio_equity.pct_change()
    asset_ret = close_wide.pct_change()

    rows = []
    total_pnl = port_pnl.sum()
    port_vol = port_ret.std() * np.sqrt(252) if port_ret.std() > 0 else np.nan

    for ticker in equity_by_asset.columns:
        pnl_i = asset_pnl[ticker].sum()
        ret_contrib = float(pnl_i / total_pnl) if total_pnl != 0 else 0.0

        w_avg = float(exec_weights[ticker].mean())
        r_i = asset_ret[ticker].dropna()
        corr_ip = r_i.corr(port_ret.reindex(r_i.index))
        vol_i = r_i.std() * np.sqrt(252) if r_i.std() > 0 else 0.0
        vol_contrib = (
            float(w_avg * vol_i * corr_ip / port_vol)
            if port_vol and not np.isnan(port_vol) and port_vol > 0
            else 0.0
        )

        cum_asset_ret = float((1 + (exec_weights[ticker].shift(0) * asset_ret[ticker]).fillna(0)).prod() - 1)
        rows.append(
            {
                "ticker": ticker,
                "pnl_usd": float(pnl_i),
                "contribution_to_return": ret_contrib,
                "contribution_to_return_pct": ret_contrib * 100,
                "avg_weight": w_avg,
                "asset_vol_annual": vol_i,
                "corr_to_portfolio": float(corr_ip) if corr_ip is not None else np.nan,
                "contribution_to_volatility": vol_contrib,
                "linked_return_proxy": cum_asset_ret,
            }
        )

    return pd.DataFrame(rows).set_index("ticker")


# --- Diagnostic counterfactual weight builders (analysis only) ---


def diagnostic_weights_momentum_only(
    momentum_wide: pd.DataFrame,
    k: int,
    max_weight: float = 0.4,
) -> pd.DataFrame:
    """
    Absolute momentum filter: momentum > 0, then top-k by raw momentum.
    No cross-sectional z-score.
    """
    tickers = list(momentum_wide.columns)
    weights = pd.DataFrame(0.0, index=momentum_wide.index, columns=tickers)
    for dt, row in momentum_wide.iterrows():
        valid = row.dropna()
        positive = valid[valid > 0]
        if positive.empty:
            continue
        picks = positive.nlargest(min(k, len(positive))).index.tolist()
        w = min(1.0 / len(picks), max_weight)
        for t in picks:
            weights.loc[dt, t] = w
        s = weights.loc[dt].sum()
        if s > 0:
            weights.loc[dt] /= s
    return weights


def diagnostic_weights_cross_sectional_only(
    momentum_wide: pd.DataFrame,
    k: int,
    max_weight: float = 0.4,
) -> pd.DataFrame:
    """Top-k by cross-sectional z-score of momentum (relative rotation, not raw level)."""
    z = cross_sectional_zscore(momentum_wide)
    w = top_k_equal_weight_weights(z, k=k, max_weight=max_weight)
    return apply_max_weight_cap(w, max_weight)


def run_counterfactual_backtest(
    label: str,
    weights: pd.DataFrame,
    open_wide: pd.DataFrame,
    close_wide: pd.DataFrame,
    vol_wide: pd.DataFrame,
    vol_feat: pd.DataFrame,
    tradable_mask: pd.Series,
    start_capital: float,
    fee_bps: float,
    slippage_vol_coef: float,
) -> dict:
    """Same execution engine; alternate weights for diagnostic comparison only."""
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
    ret = bt["equity_series"].pct_change().dropna()
    return {
        "model": label,
        "sharpe": _annualized_sharpe(ret),
        "turnover_annual": float(bt["turnover_daily"].mean() * 252),
        "max_drawdown_pct": _max_drawdown(bt["equity_series"]),
        "return_pct": bt["return_pct"],
        "trades": bt["trades"],
    }


def rolling_sharpe_series(returns: pd.Series, window: int) -> pd.Series:
    min_p = max(window // 3, 20)

    def _sharpe(x: np.ndarray) -> float:
        if len(x) < 2 or np.nanstd(x) == 0:
            return np.nan
        return float(np.nanmean(x) / np.nanstd(x) * np.sqrt(252))

    return returns.rolling(window, min_periods=min_p).apply(_sharpe, raw=True)


def edge_stability_analysis(portfolio_returns: pd.Series) -> dict:
    """Rolling 3m / 6m / 12m Sharpe — cluster vs persistent edge."""
    windows = {"3m": 63, "6m": 126, "12m": 252}
    out: dict = {}
    for label, w in windows.items():
        rs = rolling_sharpe_series(portfolio_returns, w).dropna()
        if rs.empty:
            out[label] = {"mean": np.nan, "pct_positive": np.nan, "n_clusters_positive": 0}
            continue
        positive = rs > 0
        # Count contiguous positive runs (clusters of edge)
        runs = positive.ne(positive.shift()).cumsum()
        pos_runs = positive.groupby(runs).sum()
        n_pos_clusters = int((pos_runs > 0).sum()) if len(pos_runs) else 0
        out[label] = {
            "mean_rolling_sharpe": float(rs.mean()),
            "std_rolling_sharpe": float(rs.std()),
            "pct_months_sharpe_positive": float(positive.mean()),
            "n_positive_clusters": n_pos_clusters,
            "latest_rolling_sharpe": float(rs.iloc[-1]),
        }

    rs_6m = rolling_sharpe_series(portfolio_returns, 126).dropna()
    verdict = "clustered"
    if len(rs_6m) > 20:
        pct_pos = (rs_6m > 0).mean()
        if pct_pos > 0.65 and rs_6m.std() < 1.5:
            verdict = "persistent"
        elif pct_pos < 0.45:
            verdict = "weak_or_absent"

    return {
        "rolling_windows": out,
        "edge_verdict": verdict,
        "interpretation": (
            "persistent: rolling Sharpe > 0 most of the time with moderate variance; "
            "clustered: edge concentrated in episodic windows; "
            "weak_or_absent: edge not stable"
        ),
    }


@dataclass
class EdgeDecompositionReport:
    beta_decomposition: dict
    pnl_attribution: pd.DataFrame
    model_breakdown: pd.DataFrame
    edge_stability: dict
    narrative: dict = field(default_factory=dict)


def run_edge_decomposition(
    *,
    bt: dict,
    close_wide: pd.DataFrame,
    open_wide: pd.DataFrame,
    vol_wide: pd.DataFrame,
    vol_feat: pd.DataFrame,
    momentum_wide: pd.DataFrame,
    production_weights: pd.DataFrame,
    tradable_mask: pd.Series,
    start_capital: float,
    fee_bps: float,
    slippage_vol_coef: float,
    top_k: int,
    max_weight: float,
    spy_ticker: str = "SPY",
) -> EdgeDecompositionReport:
    port_eq = bt["equity_series"]
    port_ret = port_eq.pct_change().dropna()
    spy_ret = close_wide[spy_ticker].pct_change().reindex(port_ret.index)

    beta_dec = ols_portfolio_vs_spy(port_ret, spy_ret)

    exec_w = bt["exec_weights"]
    pnl_attr = pnl_attribution_by_asset(
        bt["equity_by_asset"],
        port_eq,
        exec_w,
        close_wide,
    )

    # Counterfactual model comparison (diagnostic backtests only)
    w_mom = diagnostic_weights_momentum_only(momentum_wide, top_k, max_weight)
    w_xs = diagnostic_weights_cross_sectional_only(momentum_wide, top_k, max_weight)

    bt_kwargs = dict(
        open_wide=open_wide,
        close_wide=close_wide,
        vol_wide=vol_wide,
        vol_feat=vol_feat,
        tradable_mask=tradable_mask,
        start_capital=start_capital,
        fee_bps=fee_bps,
        slippage_vol_coef=slippage_vol_coef,
    )

    rows = [
        run_counterfactual_backtest("full_system", production_weights, **bt_kwargs),
        run_counterfactual_backtest("momentum_only", w_mom, **bt_kwargs),
        run_counterfactual_backtest("cross_sectional_only", w_xs, **bt_kwargs),
    ]
    model_breakdown = pd.DataFrame(rows)

    stability = edge_stability_analysis(port_ret)

    # Narrative answers
    beta_only = beta_dec["r_squared"] > 0.85 and abs(beta_dec["beta"]) > 0.9
    has_alpha = beta_dec["alpha_annualized"] > 0.02 and beta_dec["r_squared"] < 0.95
    mom_vs_xs = model_breakdown.set_index("model")
    rotation_edge = (
        mom_vs_xs.loc["cross_sectional_only", "sharpe"]
        > mom_vs_xs.loc["momentum_only", "sharpe"]
    ) if "cross_sectional_only" in mom_vs_xs.index else False

    narrative = {
        "question": "strategy vs market beta + momentum rotation?",
        "beta_explains_return": beta_only,
        "estimated_annual_alpha": beta_dec["alpha_annualized"],
        "has_material_alpha": has_alpha,
        "rotation_adds_vs_absolute_momentum": rotation_edge,
        "edge_stability_verdict": stability["edge_verdict"],
        "top_return_contributor": (
            pnl_attr["contribution_to_return"].idxmax()
            if len(pnl_attr)
            else None
        ),
    }

    return EdgeDecompositionReport(
        beta_decomposition=beta_dec,
        pnl_attribution=pnl_attr,
        model_breakdown=model_breakdown,
        edge_stability=stability,
        narrative=narrative,
    )


def save_edge_decomposition(
    report: EdgeDecompositionReport,
    results_dir: str | Path = "results",
) -> None:
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)

    payload = {
        "beta_decomposition": report.beta_decomposition,
        "edge_stability": report.edge_stability,
        "narrative": report.narrative,
        "pnl_attribution_summary": report.pnl_attribution.reset_index().to_dict(orient="records"),
        "model_breakdown": report.model_breakdown.to_dict(orient="records"),
    }
    with open(out / "edge_decomposition.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)

    report.pnl_attribution.to_csv(out / "pnl_attribution.csv")
    report.model_breakdown.to_csv(out / "model_breakdown.csv", index=False)
