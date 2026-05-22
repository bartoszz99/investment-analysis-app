"""
Benchmark + robustness analysis layer (STRICTLY post-hoc).
Does NOT modify signals, portfolio weights, or execution timing.

Answers:
  1) Alpha vs SPY?  — excess return, rolling beta, tracking error
  2) Stable in time? — 50/50 rank stability, regime Sharpe
  3) Leverage/momentum beta only? — beta, correlation decomposition
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


def _normalize_equity(equity: pd.Series) -> pd.Series:
    e = equity.dropna()
    if e.empty or e.iloc[0] == 0:
        return e
    return e / float(e.iloc[0])


def spy_buy_hold_equity(
    close_spy: pd.Series,
    start_capital: float,
    *,
    fee_bps: float = 0.0,
) -> pd.Series:
    """
    SPY buy & hold on aligned calendar (close-to-close MTM).
    One-way entry fee at t=1 optional; analysis-only benchmark.
    """
    close = close_spy.dropna().sort_index()
    if close.empty:
        return pd.Series(dtype=float)
    ret = close.pct_change().fillna(0.0)
    if fee_bps > 0 and len(ret) > 1:
        ret.iloc[1] -= fee_bps / 10_000.0
    equity = start_capital * (1.0 + ret).cumprod()
    equity.iloc[0] = start_capital
    return equity


def rolling_beta(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 60,
) -> pd.Series:
    """beta_t = Cov(R_p, R_SPY) / Var(R_SPY) over trailing window."""
    aligned = pd.concat(
        [portfolio_returns.rename("p"), benchmark_returns.rename("b")],
        axis=1,
    ).dropna()
    cov = aligned["p"].rolling(window, min_periods=max(20, window // 3)).cov(aligned["b"])
    var = aligned["b"].rolling(window, min_periods=max(20, window // 3)).var()
    return cov / var.replace(0, np.nan)


def rolling_correlation(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 60,
) -> pd.Series:
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    return aligned.iloc[:, 0].rolling(window, min_periods=max(20, window // 3)).corr(aligned.iloc[:, 1])


def rolling_tracking_error(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 60,
) -> pd.Series:
    """tracking_error_t = std(R_p - R_SPY) over trailing window (annualized)."""
    active = (portfolio_returns - benchmark_returns).dropna()
    te = active.rolling(window, min_periods=max(20, window // 3)).std() * np.sqrt(252)
    return te.reindex(portfolio_returns.index)


def _regime_mask_from_spy_vol(
    spy_close: pd.Series,
    vol_window: int = 20,
    high_quantile: float = 0.70,
) -> pd.Series:
    """
    High-vol regime = top 30% of trailing SPY realized vol (expanding quantile for causality).
    Analysis-only labeling; expanding threshold uses history <= t.
    """
    lagged = spy_close.shift(1)
    realized = lagged.pct_change().rolling(vol_window, min_periods=vol_window).std() * np.sqrt(252)
    threshold = realized.expanding(min_periods=vol_window).quantile(high_quantile)
    high_vol = realized >= threshold
    return high_vol.fillna(False)


def _regime_metrics(
    returns: pd.Series,
    turnover: pd.Series,
    regime_mask: pd.Series,
    label: str,
) -> dict:
    mask = regime_mask.reindex(returns.index).fillna(False)
    r = returns[mask]
    t = turnover.reindex(returns.index).fillna(0.0)[mask]
    return {
        "regime": label,
        "n_days": int(mask.sum()),
        "sharpe": _annualized_sharpe(r),
        "mean_daily_return": float(r.mean()) if len(r) else 0.0,
        "annualized_turnover": float(t.mean() * 252) if len(t) else 0.0,
        "mean_daily_turnover": float(t.mean()) if len(t) else 0.0,
    }


def split_sample_rank_stability(ranks: pd.DataFrame) -> dict:
    """
    50/50 time split (not random).
    Spearman: mean rank per asset in period 1 vs period 2.
    """
    ranks = ranks.dropna(how="all")
    if len(ranks) < 4:
        return {"spearman": np.nan, "n_assets": 0, "split": "50/50"}
    mid = len(ranks) // 2
    first = ranks.iloc[:mid]
    second = ranks.iloc[mid:]
    mean1 = first.mean(axis=0)
    mean2 = second.mean(axis=0)
    common = mean1.dropna().index.intersection(mean2.dropna().index)
    if len(common) < 2:
        return {"spearman": np.nan, "n_assets": len(common), "split": "50/50"}
    spearman = mean1[common].rank().corr(mean2[common].rank())
    return {
        "spearman": float(spearman) if spearman is not None else np.nan,
        "n_assets": len(common),
        "split": "50/50",
        "period_1": (str(first.index.min().date()), str(first.index.max().date())),
        "period_2": (str(second.index.min().date()), str(second.index.max().date())),
    }


@dataclass
class BenchmarkRobustnessReport:
    benchmark_comparison: pd.DataFrame
    benchmark_vs_strategy: pd.DataFrame
    risk_decomposition: dict
    regime_performance: dict
    stability: dict
    summary: dict = field(default_factory=dict)


def run_benchmark_robustness(
    *,
    strategy_equity: pd.Series,
    close_wide: pd.DataFrame,
    ranks: pd.DataFrame,
    turnover_daily: pd.Series,
    start_capital: float,
    benchmark_ticker: str = "SPY",
    rolling_window: int = 60,
    fee_bps: float = 0.0,
) -> BenchmarkRobustnessReport:
    """
    Post-hoc analysis vs SPY buy & hold on the same calendar.
    """
    idx = strategy_equity.index
    if benchmark_ticker not in close_wide.columns:
        raise ValueError(f"Benchmark {benchmark_ticker} not in close panel")

    spy_close = close_wide[benchmark_ticker].reindex(idx)
    spy_equity = spy_buy_hold_equity(spy_close, start_capital, fee_bps=fee_bps)
    spy_equity = spy_equity.reindex(idx).ffill()

    strat_norm = _normalize_equity(strategy_equity)
    spy_norm = _normalize_equity(spy_equity)

    strat_ret = strategy_equity.pct_change()
    spy_ret = spy_close.pct_change()
    aligned_ret = pd.concat([strat_ret.rename("strategy"), spy_ret.rename("spy")], axis=1).dropna()

    beta_roll = rolling_beta(strat_ret, spy_ret, rolling_window)
    corr_roll = rolling_correlation(strat_ret, spy_ret, rolling_window)
    te_roll = rolling_tracking_error(strat_ret, spy_ret, rolling_window)
    active_ret = strat_ret - spy_ret

    comparison = pd.DataFrame(
        {
            "strategy_equity_norm": strat_norm,
            "spy_buy_hold_norm": spy_norm.reindex(idx),
            "strategy_equity": strategy_equity,
            "spy_equity": spy_equity.reindex(idx),
            "excess_return_cum": (strat_norm - spy_norm.reindex(idx)),
            "rolling_beta_60d": beta_roll.reindex(idx),
            "rolling_corr_60d": corr_roll.reindex(idx),
            "rolling_tracking_error_60d": te_roll.reindex(idx),
        },
        index=idx,
    )

    total_days = max(len(aligned_ret), 1)
    years = total_days / 252.0
    strat_total = float(strat_norm.iloc[-1] - 1.0) if len(strat_norm) else 0.0
    spy_total = float(spy_norm.iloc[-1] - 1.0) if len(spy_norm) else 0.0

    vs_strategy = pd.DataFrame(
        [
            {
                "metric": "total_return_strategy",
                "value": strat_total,
            },
            {
                "metric": "total_return_spy",
                "value": spy_total,
            },
            {
                "metric": "excess_return_vs_spy",
                "value": strat_total - spy_total,
            },
            {
                "metric": "sharpe_strategy",
                "value": _annualized_sharpe(strat_ret),
            },
            {
                "metric": "sharpe_spy",
                "value": _annualized_sharpe(spy_ret),
            },
            {
                "metric": "sharpe_active",
                "value": _annualized_sharpe(active_ret),
            },
            {
                "metric": "mean_beta_60d",
                "value": float(beta_roll.dropna().mean()) if beta_roll.notna().any() else np.nan,
            },
            {
                "metric": "mean_correlation_60d",
                "value": float(corr_roll.dropna().mean()) if corr_roll.notna().any() else np.nan,
            },
            {
                "metric": "mean_tracking_error_60d_ann",
                "value": float(te_roll.dropna().mean()) if te_roll.notna().any() else np.nan,
            },
            {
                "metric": "information_ratio_proxy",
                "value": (
                    float(active_ret.mean() / active_ret.std() * np.sqrt(252))
                    if active_ret.std() > 0
                    else 0.0
                ),
            },
            {
                "metric": "cagr_strategy",
                "value": (float(strat_norm.iloc[-1]) ** (1 / years) - 1) if years > 0 and len(strat_norm) else 0.0,
            },
            {
                "metric": "cagr_spy",
                "value": (float(spy_norm.iloc[-1]) ** (1 / years) - 1) if years > 0 and len(spy_norm) else 0.0,
            },
        ]
    )

    regime_mask = _regime_mask_from_spy_vol(spy_close)
    regime_perf = {
        "vol_proxy": "SPY realized vol 20d (lagged close), expanding 70th pct",
        "high_volatility": _regime_metrics(
            strat_ret, turnover_daily, regime_mask, "high_volatility"
        ),
        "low_volatility": _regime_metrics(
            strat_ret, turnover_daily, ~regime_mask, "low_volatility"
        ),
        "high_volatility_spy": _regime_metrics(spy_ret, turnover_daily * 0, regime_mask, "high_volatility"),
        "low_volatility_spy": _regime_metrics(spy_ret, turnover_daily * 0, ~regime_mask, "low_volatility"),
    }

    stability = {
        "split_sample_50_50": split_sample_rank_stability(ranks),
    }

    risk_decomp = {
        "formulas": {
            "beta": "Cov(R_p, R_SPY) / Var(R_SPY)",
            "tracking_error": "std(R_p - R_SPY) annualized over rolling window",
        },
        "rolling_window": rolling_window,
        "full_sample": {
            "beta": float(
                aligned_ret["strategy"].cov(aligned_ret["spy"]) / aligned_ret["spy"].var()
            )
            if aligned_ret["spy"].var() > 0
            else np.nan,
            "correlation": float(aligned_ret["strategy"].corr(aligned_ret["spy"])),
            "tracking_error_annual": float(active_ret.dropna().std() * np.sqrt(252)),
            "active_return_mean_daily": float(active_ret.mean()),
        },
        "rolling_summary": {
            "beta_mean": float(beta_roll.dropna().mean()) if beta_roll.notna().any() else None,
            "beta_std": float(beta_roll.dropna().std()) if beta_roll.notna().any() else None,
            "correlation_mean": float(corr_roll.dropna().mean()) if corr_roll.notna().any() else None,
            "tracking_error_mean": float(te_roll.dropna().mean()) if te_roll.notna().any() else None,
        },
        "last_observation": {
            "beta_60d": float(beta_roll.dropna().iloc[-1]) if beta_roll.notna().any() else None,
            "correlation_60d": float(corr_roll.dropna().iloc[-1]) if corr_roll.notna().any() else None,
            "tracking_error_60d": float(te_roll.dropna().iloc[-1]) if te_roll.notna().any() else None,
        },
    }

    summary = {
        "alpha_vs_spy_excess_return": strat_total - spy_total,
        "beta_full_sample": risk_decomp["full_sample"]["beta"],
        "likely_momentum_beta": risk_decomp["full_sample"]["beta"] > 0.8
        and risk_decomp["full_sample"]["correlation"] > 0.7,
        "rank_stability_50_50": stability["split_sample_50_50"]["spearman"],
    }

    return BenchmarkRobustnessReport(
        benchmark_comparison=comparison,
        benchmark_vs_strategy=vs_strategy,
        risk_decomposition=risk_decomp,
        regime_performance=regime_perf,
        stability=stability,
        summary=summary,
    )


def save_benchmark_robustness(
    report: BenchmarkRobustnessReport,
    results_dir: str | Path = "results",
) -> None:
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)

    report.benchmark_comparison.to_csv(out / "benchmark_comparison.csv")
    report.benchmark_vs_strategy.to_csv(out / "benchmark_vs_strategy.csv", index=False)

    with open(out / "risk_decomposition.json", "w", encoding="utf-8") as fh:
        json.dump(report.risk_decomposition, fh, indent=2, default=str)

    regime_payload = {
        **report.regime_performance,
        "stability": report.stability,
        "summary": report.summary,
    }
    with open(out / "regime_performance.json", "w", encoding="utf-8") as fh:
        json.dump(regime_payload, fh, indent=2, default=str)
