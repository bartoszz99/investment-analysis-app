"""
Factor neutralization — STRICTLY diagnostic (post-hoc).
Removes market, sector ETF, and momentum exposures from realized portfolio returns.
Does NOT modify signals, portfolio weights, or execution.

Question answered:
  "Am I earning from US market structure, or from the model?"
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


def _annualized_sharpe(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) < 2 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(252))


def _max_drawdown_pct(cum_equity: pd.Series) -> float:
    peak = cum_equity.cummax()
    dd = (cum_equity - peak) / peak.replace(0, np.nan)
    return float(abs(dd.min()) * 100) if len(dd) else 0.0


def _ols_multivariate(
    y: pd.Series,
    factors: pd.DataFrame,
) -> tuple[pd.Series, dict]:
    """
    y ~ factors + const
    Returns (residuals aligned to y index, metadata dict).
    """
    data = pd.concat([y.rename("y"), factors], axis=1).dropna()
    if len(data) < max(10, factors.shape[1] + 5):
        return pd.Series(dtype=float), {"alpha_daily": np.nan, "betas": {}, "r_squared": np.nan}

    yv = data["y"].to_numpy()
    X = np.column_stack([np.ones(len(data)), data[factors.columns].to_numpy()])
    names = ["const"] + list(factors.columns)
    coeffs, _, _, _ = np.linalg.lstsq(X, yv, rcond=None)
    y_hat = X @ coeffs
    resid = yv - y_hat
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((yv - yv.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    betas = {names[i]: float(coeffs[i]) for i in range(len(names))}
    residuals = pd.Series(resid, index=data.index, name="residual")

    meta = {
        "alpha_daily": betas.get("const", np.nan),
        "alpha_annualized": betas.get("const", np.nan) * 252,
        "betas": {k: v for k, v in betas.items() if k != "const"},
        "r_squared": r2,
        "n_obs": len(data),
        "residual_vol_annual": float(np.std(resid) * np.sqrt(252)),
    }
    return residuals, meta


def _hedged_returns(portfolio_returns: pd.Series, factors: pd.DataFrame, betas: dict) -> pd.Series:
    """Beta-hedged returns: R_p - sum(beta_k * F_k). Preserves intercept as drift (alpha)."""
    aligned = pd.concat([portfolio_returns.rename("p"), factors], axis=1).dropna()
    hedged = aligned["p"].copy()
    for col in factors.columns:
        b = betas.get(col, 0.0)
        hedged = hedged - b * aligned[col]
    return hedged.rename("hedged")


def _residual_metrics(
    residuals: pd.Series,
    hedged: pd.Series | None = None,
) -> dict:
    if residuals.empty:
        return {
            "alpha_annualized": np.nan,
            "residual_sharpe": np.nan,
            "residual_drawdown_pct": np.nan,
            "cumulative_residual_return": np.nan,
        }
    cum = (1.0 + residuals).cumprod()
    h = hedged.dropna() if hedged is not None else residuals
    return {
        "alpha_annualized": float(residuals.mean() * 252),
        "residual_sharpe": _annualized_sharpe(h),
        "residual_drawdown_pct": _max_drawdown_pct((1.0 + h).cumprod()),
        "cumulative_residual_return": float(cum.iloc[-1] - 1.0),
    }


def build_momentum_factor_return(
    close_wide: pd.DataFrame,
    momentum_wide: pd.DataFrame,
    *,
    leg_frac: float = 0.5,
) -> pd.Series:
    """
    Diagnostic momentum factor: long top half / short bottom half by lagged momentum.
    Uses momentum at t-1 to sort; applies close-to-close returns at t (post-hoc factor).
    """
    asset_ret = close_wide.pct_change()
    mom_lag = momentum_wide.shift(1)
    factor = pd.Series(np.nan, index=asset_ret.index)

    for dt in asset_ret.index:
        m = mom_lag.loc[dt].dropna()
        r = asset_ret.loc[dt].dropna()
        common = m.index.intersection(r.index)
        if len(common) < 4:
            continue
        m = m[common]
        r = r[common]
        n = max(1, int(len(common) * leg_frac))
        longs = m.nlargest(n).index
        shorts = m.nsmallest(n).index
        factor.loc[dt] = r[longs].mean() - r[shorts].mean()

    return factor.rename("R_momentum_factor")


def rolling_momentum_exposure(
    portfolio_returns: pd.Series,
    momentum_factor: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Rolling beta of portfolio returns to momentum factor (diagnostic)."""
    aligned = pd.concat(
        [portfolio_returns.rename("p"), momentum_factor.rename("m")],
        axis=1,
    ).dropna()
    if len(aligned) < window:
        return pd.Series(dtype=float)

    cov = aligned["p"].rolling(window, min_periods=max(10, window // 2)).cov(aligned["m"])
    var = aligned["m"].rolling(window, min_periods=max(10, window // 2)).var()
    return (cov / var.replace(0, np.nan)).rename("momentum_beta_20d")


@dataclass
class FactorNeutralizationReport:
    market_neutral: dict
    sector_neutral: dict
    momentum_neutral: dict
    full_neutral: dict
    summary: dict
    rolling_momentum_beta: pd.Series = field(repr=False)
    residual_series: dict = field(default_factory=dict, repr=False)


def run_factor_neutralization(
    portfolio_returns: pd.Series,
    close_wide: pd.DataFrame,
    momentum_wide: pd.DataFrame,
    *,
    market_ticker: str = "SPY",
    sector_tickers: tuple[str, ...] = ("XLK", "XLF", "XLE"),
) -> FactorNeutralizationReport:
    """
    Sequential / nested factor removal on realized portfolio returns.
    """
    asset_ret = close_wide.pct_change()
    r_spy = asset_ret[market_ticker].rename("R_SPY")
    r_sectors = asset_ret[list(sector_tickers)].add_prefix("R_")

    mom_factor = build_momentum_factor_return(close_wide, momentum_wide)

    # 1) Market neutral: R_p ~ R_SPY
    resid_market, meta_market = _ols_multivariate(portfolio_returns, r_spy.to_frame())
    hedged_market = _hedged_returns(portfolio_returns, r_spy.to_frame(), meta_market.get("betas", {}))
    metrics_market = _residual_metrics(resid_market, hedged_market)
    metrics_market["alpha_annualized"] = meta_market.get("alpha_annualized", np.nan)

    # 2) Sector neutral: R_p ~ R_SPY + R_XLK + R_XLF + R_XLE
    sector_factors = pd.concat([r_spy, r_sectors], axis=1)
    resid_sector, meta_sector = _ols_multivariate(portfolio_returns, sector_factors)
    hedged_sector = _hedged_returns(portfolio_returns, sector_factors, meta_sector.get("betas", {}))
    metrics_sector = _residual_metrics(resid_sector, hedged_sector)
    metrics_sector["alpha_annualized"] = meta_sector.get("alpha_annualized", np.nan)

    # 3) Momentum exposure control: R_p ~ R_momentum_factor (rolling tracked separately)
    resid_mom, meta_mom = _ols_multivariate(portfolio_returns, mom_factor.to_frame())
    hedged_mom = _hedged_returns(portfolio_returns, mom_factor.to_frame(), meta_mom.get("betas", {}))
    metrics_mom = _residual_metrics(resid_mom, hedged_mom)
    roll_mom_beta = rolling_momentum_exposure(portfolio_returns, mom_factor, window=20)

    # 4) Full: market + sectors + momentum
    full_factors = pd.concat([sector_factors, mom_factor], axis=1)
    resid_full, meta_full = _ols_multivariate(portfolio_returns, full_factors)
    hedged_full = _hedged_returns(portfolio_returns, full_factors, meta_full.get("betas", {}))
    metrics_full = _residual_metrics(resid_full, hedged_full)
    metrics_full["alpha_annualized"] = meta_full.get("alpha_annualized", np.nan)

    # OLS intercept = net alpha; residual mean ≈ 0 by construction
    alpha_net_market = float(meta_market.get("alpha_annualized", np.nan))
    alpha_net_market_sector = float(meta_sector.get("alpha_annualized", np.nan))
    alpha_net_full = float(meta_full.get("alpha_annualized", np.nan))

    # Model vs structure verdict
    own_model = (
        alpha_net_market_sector > 0.02
        and metrics_sector["residual_sharpe"] > 0.3
        and meta_sector.get("r_squared", 0) < 0.95
    )
    structure_driven = meta_sector.get("r_squared", 0) > 0.75 and alpha_net_market_sector < 0.02

    summary = {
        "question": "US market structure vs own model?",
        "alpha_net_market": alpha_net_market,
        "alpha_net_market_sector": alpha_net_market_sector,
        "alpha_net_full_factors": alpha_net_full,
        "residual_sharpe_market_neutral": metrics_market["residual_sharpe"],
        "residual_sharpe_sector_neutral": metrics_sector["residual_sharpe"],
        "residual_sharpe_full_neutral": metrics_full["residual_sharpe"],
        "residual_drawdown_market_neutral_pct": metrics_market["residual_drawdown_pct"],
        "residual_drawdown_sector_neutral_pct": metrics_sector["residual_drawdown_pct"],
        "residual_drawdown_full_neutral_pct": metrics_full["residual_drawdown_pct"],
        "mean_momentum_beta_20d": float(roll_mom_beta.dropna().mean())
        if roll_mom_beta.notna().any()
        else np.nan,
        "verdict_own_model": own_model,
        "verdict_structure_driven": structure_driven,
        "interpretation": (
            "own_model=True: meaningful alpha after SPY+sector factors; "
            "structure_driven=True: returns largely explained by US beta/sector/momentum factors"
        ),
    }

    return FactorNeutralizationReport(
        market_neutral={**meta_market, **metrics_market, "label": "R_p ~ R_SPY"},
        sector_neutral={**meta_sector, **metrics_sector, "label": "R_p ~ R_SPY + XLK + XLF + XLE"},
        momentum_neutral={
            **meta_mom,
            **metrics_mom,
            "label": "R_p ~ R_momentum_factor",
            "rolling_20d_beta_mean": summary["mean_momentum_beta_20d"],
        },
        full_neutral={
            **meta_full,
            **metrics_full,
            "label": "R_p ~ R_SPY + sectors + momentum",
        },
        summary=summary,
        rolling_momentum_beta=roll_mom_beta,
        residual_series={
            "market_neutral": resid_market,
            "market_hedged": hedged_market,
            "sector_neutral": resid_sector,
            "sector_hedged": hedged_sector,
            "momentum_neutral": resid_mom,
            "full_neutral": resid_full,
            "full_hedged": hedged_full,
        },
    )


def save_factor_neutralization(
    report: FactorNeutralizationReport,
    results_dir: str | Path = "results",
) -> None:
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)

    payload = {
        "summary": report.summary,
        "market_neutral": {k: v for k, v in report.market_neutral.items() if k != "betas"},
        "market_neutral_betas": report.market_neutral.get("betas", {}),
        "sector_neutral": {k: v for k, v in report.sector_neutral.items() if k != "betas"},
        "sector_neutral_betas": report.sector_neutral.get("betas", {}),
        "momentum_neutral": {k: v for k, v in report.momentum_neutral.items() if k != "betas"},
        "momentum_neutral_betas": report.momentum_neutral.get("betas", {}),
        "full_neutral": {k: v for k, v in report.full_neutral.items() if k != "betas"},
        "full_neutral_betas": report.full_neutral.get("betas", {}),
    }
    with open(out / "factor_neutralization.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)

    resid_df = pd.DataFrame(report.residual_series)
    resid_df.to_csv(out / "factor_neutral_returns.csv")
