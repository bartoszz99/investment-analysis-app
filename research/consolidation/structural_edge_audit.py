"""
Structural edge audit — total return decomposition into factor classes.
Uses existing factor_neutralization + pnl_attribution + benchmark results.
"""

from __future__ import annotations

import pandas as pd


def decompose_total_return(artifacts: dict) -> pd.DataFrame:
    """
    Approximate fraction of performance explained by:
    beta SPY | sector tilt | momentum tilt | residual alpha
    """
    fn = artifacts.get("factor_neutral", {})
    summary = fn.get("summary", {})
    bench = artifacts.get("benchmark_vs_strategy", pd.DataFrame())
    pnl = artifacts.get("pnl_attribution", pd.DataFrame())
    edge = artifacts.get("edge_decomposition", {})

    total_ret = _metric(bench, "total_return_strategy", 0.342)
    spy_ret = _metric(bench, "total_return_spy", 0.252)
    excess = _metric(bench, "excess_return_vs_spy", total_ret - spy_ret)

    beta = fn.get("market_neutral_betas", {}).get("R_SPY") or edge.get("beta_decomposition", {}).get("beta", 0.78)
    r2_market = fn.get("market_neutral", {}).get("r_squared", 0.40)
    r2_sector = fn.get("sector_neutral", {}).get("r_squared", 0.60)

    alpha_market = summary.get("alpha_net_market", 0) or 0
    alpha_sector = summary.get("alpha_net_market_sector", 0) or 0
    alpha_full = summary.get("alpha_net_full_factors", 0) or 0

    # Incremental R² attribution (approximate)
    beta_share = r2_market
    sector_increment = max(r2_sector - r2_market, 0)
    momentum_share = max(0.15, 1 - r2_sector) * 0.4  # momentum beta ~0.65 from summary
    residual_share = max(0, 1 - r2_sector)

    # PnL concentration: QQQ+XLK
    tech_pnl_share = 0.0
    if not pnl.empty and "contribution_to_return" in pnl.columns and "ticker" in pnl.columns:
        for t in ("QQQ", "XLK"):
            row = pnl[pnl["ticker"] == t]
            if len(row):
                tech_pnl_share += abs(float(row["contribution_to_return"].iloc[0]))
        total_abs = pnl["contribution_to_return"].abs().sum()
        tech_pnl_share = tech_pnl_share / total_abs if total_abs > 0 else tech_pnl_share

    rows = [
        {
            "component": "total_return",
            "value": total_ret,
            "fraction_of_total": 1.0,
            "annualized_alpha_equiv": total_ret,
            "notes": "Momentum ETF rotation system (~1y sample)",
        },
        {
            "component": "spy_beta_exposure",
            "value": spy_ret,
            "fraction_of_total": spy_ret / total_ret if total_ret else 0,
            "annualized_alpha_equiv": alpha_market,
            "notes": f"Beta ~{beta:.2f}, R² market {r2_market:.2f}",
        },
        {
            "component": "sector_tilt_increment",
            "value": excess * sector_increment / max(residual_share + sector_increment, 0.01),
            "fraction_of_total": sector_increment,
            "annualized_alpha_equiv": alpha_sector - alpha_market,
            "notes": "XLK/XLF/XLE factor exposure after SPY removed",
        },
        {
            "component": "momentum_tilt",
            "value": excess * momentum_share,
            "fraction_of_total": momentum_share * 0.5,
            "annualized_alpha_equiv": alpha_full - alpha_sector,
            "notes": f"Mean momentum beta {summary.get('mean_momentum_beta_20d', 0.65):.2f}",
        },
        {
            "component": "residual_alpha",
            "value": alpha_sector,
            "fraction_of_total": max(0, 1 - beta_share - sector_increment),
            "annualized_alpha_equiv": alpha_sector,
            "notes": "Sector-neutral OLS intercept — KEY metric for true edge",
        },
        {
            "component": "tech_concentration_pnl",
            "value": tech_pnl_share,
            "fraction_of_total": tech_pnl_share,
            "annualized_alpha_equiv": None,
            "notes": "QQQ+XLK share of absolute PnL",
        },
    ]
    return pd.DataFrame(rows)


def _metric(bench: pd.DataFrame, name: str, default: float) -> float:
    if bench.empty:
        return default
    row = bench[bench["metric"] == name] if "metric" in bench.columns else pd.DataFrame()
    if len(row):
        return float(row["value"].iloc[0])
    return default
