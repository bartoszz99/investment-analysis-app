"""
Robustness & Fragility Analysis — diagnostic orchestrator.
Does NOT modify production signals, execution, portfolio, benchmark, or neutralization.

Answers: real edge vs tech bull overlay vs curve-fit?
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from analysis.bootstrap_fragility import block_bootstrap
from analysis.parameter_sensitivity import run_parameter_grid
from analysis.robustness_common import annualized_sharpe, ols_alpha_beta
from analysis.subperiod_analysis import run_subperiod_analysis


def _fetch_series(ticker: str, calendar: pd.DatetimeIndex, period: str) -> pd.Series:
    hist = yf.Ticker(ticker).history(period=period)
    if hist.empty:
        return pd.Series(np.nan, index=calendar)
    s = hist["Close"].copy()
    if s.index.tz:
        s.index = s.index.tz_localize(None)
    s.index = pd.DatetimeIndex([pd.Timestamp(d).normalize() for d in s.index])
    return s.reindex(calendar)


def market_concentration_analysis(equity_by_asset: pd.DataFrame) -> dict:
    """PnL concentration across ETFs (diagnostic)."""
    pnl = equity_by_asset.diff()
    total = pnl.sum(axis=1).replace(0, np.nan)
    share = pnl.div(total, axis=0).abs()

    qqq_share = share.get("QQQ", pd.Series(0.0, index=pnl.index)).fillna(0)
    xlk_share = share.get("XLK", pd.Series(0.0, index=pnl.index)).fillna(0)
    tech_share = qqq_share + xlk_share

    cum_pnl = pnl.sum()
    total_abs = cum_pnl.abs().sum()
    qqq_pnl_pct = float(cum_pnl.get("QQQ", 0) / total_abs) if total_abs else 0.0
    xlk_pnl_pct = float(cum_pnl.get("XLK", 0) / total_abs) if total_abs else 0.0
    others = [c for c in pnl.columns if c not in ("QQQ", "XLK")]
    others_pnl_pct = float(cum_pnl[others].sum() / total_abs) if total_abs and others else 0.0

    avg_w = (equity_by_asset / equity_by_asset.sum(axis=1).replace(0, np.nan).values[:, None])
    hhi = float((avg_w.fillna(0) ** 2).sum(axis=1).mean())

    max_single = share.max(axis=1)
    pct_days_80 = float((max_single > 0.80).mean())

    tech_dom = (qqq_pnl_pct + xlk_pnl_pct) > 0.65
    verdict = "tech_concentration" if tech_dom else "diversified"

    return {
        "qqq_pnl_share": qqq_pnl_pct,
        "xlk_pnl_share": xlk_pnl_pct,
        "others_pnl_share": others_pnl_pct,
        "concentration_ratio_top2": float(qqq_pnl_pct + xlk_pnl_pct),
        "herfindahl_index": hhi,
        "pct_days_single_etf_gt_80pct_pnl": pct_days_80,
        "verdict": verdict,
    }


def regime_dependency_analysis(
    equity: pd.Series,
    close_wide: pd.DataFrame,
    period: str,
) -> dict:
    """
    Performance conditional on macro/regime labels (post-hoc, expanding thresholds).
    """
    ret = equity.pct_change().dropna()
    cal = equity.index
    spy = close_wide["SPY"].reindex(cal)
    xlk = close_wide["XLK"].reindex(cal)

    vix = _fetch_series("^VIX", cal, period)
    tnx = _fetch_series("^TNX", cal, period)

    spy_lag = spy.shift(1)
    sma200 = spy_lag.rolling(200, min_periods=60).mean()
    bullish = spy_lag > sma200

    vix_lag = vix.shift(1)
    vix_thresh = vix_lag.expanding(min_periods=20).quantile(0.70)
    vix_high = vix_lag >= vix_thresh

    tnx_chg = tnx.shift(1).pct_change()
    rates_rising = tnx_chg > 0

    rs = (xlk.shift(1) / spy_lag).pct_change(20)
    rs_thresh = rs.expanding(min_periods=20).median()
    tech_leading = rs >= rs_thresh

    def _sharpe_in(mask: pd.Series) -> float:
        m = mask.reindex(ret.index).fillna(False)
        return annualized_sharpe(ret[m])

    return {
        "sharpe_vix_high": _sharpe_in(vix_high),
        "sharpe_vix_low": _sharpe_in(~vix_high),
        "sharpe_spy_bullish": _sharpe_in(bullish),
        "sharpe_spy_bearish": _sharpe_in(~bullish),
        "sharpe_rates_rising": _sharpe_in(rates_rising),
        "sharpe_rates_falling": _sharpe_in(~rates_rising),
        "sharpe_tech_leading": _sharpe_in(tech_leading),
        "sharpe_tech_lagging": _sharpe_in(~tech_leading),
        "regime_dependency_score": _regime_dependency_score(
            _sharpe_in(vix_high),
            _sharpe_in(~vix_high),
            _sharpe_in(bullish),
            _sharpe_in(tech_leading),
        ),
        "works_only_low_vol": _sharpe_in(~vix_high) > max(_sharpe_in(vix_high), 0) * 2,
        "works_only_bull": _sharpe_in(bullish) > max(_sharpe_in(~bullish), 0) * 2,
        "works_only_tech_leadership": _sharpe_in(tech_leading) > max(_sharpe_in(~tech_leading), 0) * 2,
    }


def _regime_dependency_score(*sharpes: float) -> float:
    arr = np.array([s for s in sharpes if s == s])
    if len(arr) < 2:
        return 0.0
    return float(arr.max() - arr.min())


def classify_edge_quality(
    *,
    fragility: float,
    subperiod_verdict: str,
    bootstrap_survival: float,
    tech_concentration: bool,
    residual_alpha_sector: float,
    likely_overfit: bool,
) -> str:
    if likely_overfit or subperiod_verdict == "single_regime_only":
        return "curve_fit_or_single_regime"
    if tech_concentration and residual_alpha_sector <= 0:
        return "momentum_overlay_tech_bull"
    if fragility < 0.5 and bootstrap_survival > 0.6 and subperiod_verdict == "persistent":
        return "potential_real_edge"
    if residual_alpha_sector > 0.02 and bootstrap_survival > 0.5:
        return "modest_alpha_requires_validation"
    return "uncertain_mixed_signals"


def build_final_verdict(
    *,
    param_surface: pd.DataFrame,
    subperiod_summary: dict,
    bootstrap_summary: dict,
    concentration: dict,
    regime_dep: dict,
    factor_neutral_summary: dict | None,
) -> dict:
    fragility = float(param_surface["fragility_score_global"].iloc[0]) if len(param_surface) else np.nan
    if fragility != fragility:
        sharpe_std = param_surface["sharpe"].std()
        sharpe_mean = param_surface["sharpe"].mean()
        fragility = float(sharpe_std / abs(sharpe_mean)) if abs(sharpe_mean) > 1e-6 else float(sharpe_std)

    high_fragility = fragility > 1.0 if fragility == fragility else True
    bootstrap_survival = bootstrap_summary.get("bootstrap_survival_rate", 0.0)
    weak_bootstrap = bootstrap_survival < 0.45
    sub_verdict = subperiod_summary.get("verdict", "unstable")
    single_regime = sub_verdict == "single_regime_only"

    residual_alpha = 0.0
    sector_collapse = False
    if factor_neutral_summary:
        residual_alpha = factor_neutral_summary.get("alpha_net_market_sector", 0.0)
        sector_collapse = residual_alpha <= 0

    tech_conc = concentration.get("verdict") == "tech_concentration"
    likely_overfit = high_fragility and (single_regime or weak_bootstrap)

    production_candidate = not any(
        [
            high_fragility,
            weak_bootstrap,
            single_regime,
            sector_collapse,
            likely_overfit,
        ]
    )

    edge_quality = classify_edge_quality(
        fragility=fragility if fragility == fragility else 999,
        subperiod_verdict=sub_verdict,
        bootstrap_survival=bootstrap_survival,
        tech_concentration=tech_conc,
        residual_alpha_sector=residual_alpha,
        likely_overfit=likely_overfit,
    )

    narrative = _narrative(edge_quality, tech_conc, single_regime, high_fragility, bootstrap_survival)

    return {
        "edge_quality": edge_quality,
        "fragility_score": fragility,
        "persistent_across_subperiods": sub_verdict == "persistent",
        "subperiod_verdict": sub_verdict,
        "bootstrap_survival_rate": bootstrap_survival,
        "prob_negative_sharpe": bootstrap_summary.get("prob_negative_sharpe"),
        "prob_underperform_spy": bootstrap_summary.get("prob_underperform_spy"),
        "left_tail_loss_95": bootstrap_summary.get("left_tail_loss_95"),
        "tech_concentration": tech_conc,
        "concentration_verdict": concentration.get("verdict"),
        "likely_overfit": likely_overfit,
        "market_regime_dependency": regime_dep,
        "residual_alpha_sector_neutral": residual_alpha,
        "production_candidate": production_candidate,
        "narrative": narrative,
    }


def _narrative(
    edge_quality: str,
    tech_conc: bool,
    single_regime: bool,
    high_fragility: bool,
    bootstrap_survival: float,
) -> str:
    parts = []
    if edge_quality == "potential_real_edge":
        parts.append("Structure appears relatively stable across params and subperiods.")
    elif edge_quality == "momentum_overlay_tech_bull":
        parts.append("Returns look like momentum overlay on tech bull market (QQQ/XLK dominated).")
    elif edge_quality == "curve_fit_or_single_regime":
        parts.append("Likely curve-fit or single-regime luck — not robust.")
    else:
        parts.append("Mixed/uncertain — requires longer sample and OOS validation.")

    if tech_conc:
        parts.append("PnL concentrated in QQQ/XLK.")
    if single_regime:
        parts.append("Most PnL from one subperiod.")
    if high_fragility:
        parts.append("High parameter sensitivity (fragility).")
    if bootstrap_survival < 0.5:
        parts.append(f"Bootstrap survival weak ({bootstrap_survival:.0%}).")
    return " ".join(parts)


@dataclass
class RobustnessFragilityReport:
    parameter_surface: pd.DataFrame
    subperiod_metrics: pd.DataFrame
    subperiod_summary: dict
    bootstrap_distribution: pd.DataFrame
    bootstrap_summary: dict
    concentration: dict
    regime_dependency: dict
    final_verdict: dict


def run_robustness_fragility_analysis(
    *,
    bt: dict,
    feature_panels: dict,
    open_wide: pd.DataFrame,
    close_wide: pd.DataFrame,
    vol_wide: pd.DataFrame,
    vol_feat: pd.DataFrame,
    tradable_mask: pd.Series,
    start_capital: float,
    fee_bps: float,
    slippage_vol_coef: float,
    max_weight: float,
    period: str,
    factor_neutral_summary: dict | None = None,
    bootstrap_samples: int = 1000,
) -> RobustnessFragilityReport:
    equity = bt["equity_series"]
    turnover = bt["turnover_daily"]
    spy_close = close_wide["SPY"]

    param_surface = run_parameter_grid(
        feature_panels=feature_panels,
        open_wide=open_wide,
        close_wide=close_wide,
        vol_wide=vol_wide,
        vol_feat=vol_feat,
        tradable_mask=tradable_mask,
        start_capital=start_capital,
        fee_bps=fee_bps,
        slippage_vol_coef=slippage_vol_coef,
        max_weight=max_weight,
    )

    subperiod_df, sub_summary = run_subperiod_analysis(equity, turnover, spy_close)

    port_ret = equity.pct_change().dropna()
    spy_ret = spy_close.pct_change()
    boot_dist, boot_summary = block_bootstrap(
        port_ret,
        spy_ret,
        n_samples=bootstrap_samples,
    )

    concentration = market_concentration_analysis(bt["equity_by_asset"])
    regime_dep = regime_dependency_analysis(equity, close_wide, period)

    final = build_final_verdict(
        param_surface=param_surface,
        subperiod_summary=sub_summary,
        bootstrap_summary=boot_summary,
        concentration=concentration,
        regime_dep=regime_dep,
        factor_neutral_summary=factor_neutral_summary,
    )

    return RobustnessFragilityReport(
        parameter_surface=param_surface,
        subperiod_metrics=subperiod_df,
        subperiod_summary=sub_summary,
        bootstrap_distribution=boot_dist,
        bootstrap_summary=boot_summary,
        concentration=concentration,
        regime_dependency=regime_dep,
        final_verdict=final,
    )


def save_robustness_fragility(
    report: RobustnessFragilityReport,
    results_dir: str | Path = "results",
) -> None:
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)

    payload = {
        **report.final_verdict,
        "subperiod_summary": report.subperiod_summary,
        "bootstrap_summary": report.bootstrap_summary,
        "concentration": report.concentration,
        "regime_dependency": {
            k: v for k, v in report.regime_dependency.items() if not isinstance(v, (dict, list))
        },
    }
    with open(out / "robustness_summary.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)

    report.parameter_surface.to_csv(out / "parameter_surface.csv", index=False)
    report.subperiod_metrics.to_csv(out / "subperiod_metrics.csv", index=False)
    if not report.bootstrap_distribution.empty:
        report.bootstrap_distribution.to_csv(out / "bootstrap_distribution.csv", index=False)
