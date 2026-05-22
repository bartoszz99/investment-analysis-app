"""
Multi-asset ETF backtest runner (US universe).
Anti-leakage: per-ticker features; cross-section uses same-day scores only;
execution lag shift(1) -> Open[t+1].
main.py single-asset SMA/B&H unchanged — run this script separately.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import config
from analysis.benchmark_robustness import run_benchmark_robustness, save_benchmark_robustness
from analysis.edge_decomposition import run_edge_decomposition, save_edge_decomposition
from analysis.factor_neutralization import run_factor_neutralization, save_factor_neutralization
from analysis.robustness_analysis import run_robustness_fragility_analysis, save_robustness_fragility
from layers.cross_sectional import cross_sectional_rank, cross_sectional_zscore
from layers.data_layer import load_multi_asset
from layers.multi_asset_backtest import run_multi_asset_backtest
from layers.multi_asset_features import MultiAssetFeatureEngine
from layers.multi_asset_signals import build_momentum_portfolio_weights, signal_matrix_from_weights
from validation.multi_asset_validation import (
    portfolio_validation_summary,
    save_turnover_report,
)


RESULTS_DIR = Path("results")


def _save_artifacts(
    bt: dict,
    close_wide: pd.DataFrame,
    ranks: pd.DataFrame,
    validation: dict,
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    eq = bt["equity_series"].to_frame("portfolio_equity")
    for col in bt["equity_by_asset"].columns:
        eq[f"equity_{col}"] = bt["equity_by_asset"][col]
    eq.to_csv(RESULTS_DIR / "multi_asset_equity.csv")

    contrib = bt["equity_by_asset"].copy()
    contrib["portfolio"] = bt["equity_series"].values[: len(contrib)]
    contrib.to_csv(RESULTS_DIR / "asset_contributions.csv")

    turnover_payload = {
        "turnover": validation.get("turnover", {}),
        "ranking_stability": validation.get("ranking_stability", {}),
        "deflated_sharpe": validation.get("deflated_sharpe", {}),
        "trades": bt.get("trades", 0),
        "return_pct": bt.get("return_pct", 0),
        "max_drawdown": bt.get("max_drawdown", 0),
    }
    save_turnover_report(turnover_payload, RESULTS_DIR / "turnover_report.json")

    with open(RESULTS_DIR / "correlation_matrix.json", "w", encoding="utf-8") as fh:
        json.dump(validation.get("correlation_matrix", {}), fh, indent=2)


def run_multi_asset_pipeline(period: str | None = None) -> dict:
    period = period or config.MULTI_ASSET_PERIOD
    tickers = list(config.ETF_UNIVERSE)

    print(f"\n=== Multi-asset ETF backtest ({period}) ===")
    print(f"Universe: {', '.join(tickers)}")

    bundle = load_multi_asset(tickers, period, forward_fill_alignment=False)
    tradable = bundle.drop_non_tradable()
    print(f"Calendar: {tradable.calendar.min().date()} -> {tradable.calendar.max().date()}")
    print(f"Sessions (tradable): {len(tradable.calendar)}")

    feature_engine = MultiAssetFeatureEngine()
    feature_panels = feature_engine.build_panel(tradable)

    close_wide = tradable.wide("Close")
    open_wide = tradable.wide("Open")
    vol_wide = tradable.wide("Volume")
    vol_feat = feature_engine.to_wide(feature_panels, "volatility_20d")
    mom_wide = feature_engine.to_wide(feature_panels, "momentum_20d")

    weights, z_mom, ranks = build_momentum_portfolio_weights(
        feature_panels,
        top_k=config.MOMENTUM_TOP_K,
        short_enabled=config.SHORT_ENABLED,
        max_weight=config.MAX_WEIGHT_PER_ASSET,
        method=config.PORTFOLIO_METHOD,
        tradable_mask=tradable.tradable_mask(),
    )
    signals = signal_matrix_from_weights(weights)

    print("\n--- Cross-sectional snapshot (last 5 days, momentum z-score) ---")
    z = cross_sectional_zscore(mom_wide)
    print(z.tail())
    print("\n--- Ranks (1=best momentum) ---")
    print(cross_sectional_rank(mom_wide, ascending=False).tail())
    print("\n--- Target weights ---")
    print(weights.tail())

    bt = run_multi_asset_backtest(
        open_wide,
        close_wide,
        weights,
        volume=vol_wide,
        vol_panel=vol_feat,
        start_capital=config.MULTI_ASSET_START_CAPITAL,
        fee_bps=config.TRADE_FEE_BPS,
        slippage_vol_coef=config.SLIPPAGE_VOL_COEF,
        tradable_mask=tradable.tradable_mask(),
    )

    port_ret = bt["equity_series"].pct_change().dropna()
    validation = portfolio_validation_summary(
        close_wide,
        ranks,
        bt["turnover_daily"],
        port_ret,
        n_trials=len(tickers),
    )

    _save_artifacts(bt, close_wide, ranks, validation)

    # --- Analysis layer only (no signal / execution changes) ---
    robustness = run_benchmark_robustness(
        strategy_equity=bt["equity_series"],
        close_wide=close_wide,
        ranks=ranks,
        turnover_daily=bt["turnover_daily"],
        start_capital=config.MULTI_ASSET_START_CAPITAL,
        benchmark_ticker="SPY",
        rolling_window=60,
    )
    save_benchmark_robustness(robustness, RESULTS_DIR)

    edge = run_edge_decomposition(
        bt=bt,
        close_wide=close_wide,
        open_wide=open_wide,
        vol_wide=vol_wide,
        vol_feat=vol_feat,
        momentum_wide=mom_wide,
        production_weights=weights,
        tradable_mask=tradable.tradable_mask(),
        start_capital=config.MULTI_ASSET_START_CAPITAL,
        fee_bps=config.TRADE_FEE_BPS,
        slippage_vol_coef=config.SLIPPAGE_VOL_COEF,
        top_k=config.MOMENTUM_TOP_K,
        max_weight=config.MAX_WEIGHT_PER_ASSET,
    )
    save_edge_decomposition(edge, RESULTS_DIR)

    factor_neutral = run_factor_neutralization(
        portfolio_returns=bt["equity_series"].pct_change().dropna(),
        close_wide=close_wide,
        momentum_wide=mom_wide,
    )
    save_factor_neutralization(factor_neutral, RESULTS_DIR)

    # --- Robustness & fragility (diagnostic counterfactuals; production path unchanged) ---
    fragility_report = run_robustness_fragility_analysis(
        bt=bt,
        feature_panels=feature_panels,
        open_wide=open_wide,
        close_wide=close_wide,
        vol_wide=vol_wide,
        vol_feat=vol_feat,
        tradable_mask=tradable.tradable_mask(),
        start_capital=config.MULTI_ASSET_START_CAPITAL,
        fee_bps=config.TRADE_FEE_BPS,
        slippage_vol_coef=config.SLIPPAGE_VOL_COEF,
        max_weight=config.MAX_WEIGHT_PER_ASSET,
        period=period,
        factor_neutral_summary=factor_neutral.summary,
        bootstrap_samples=1000,
    )
    save_robustness_fragility(fragility_report, RESULTS_DIR)

    fv = fragility_report.final_verdict
    print("\n=== Robustness & Fragility ===")
    print(f"Edge quality:           {fv['edge_quality']}")
    print(f"Fragility score:        {fv['fragility_score']:.3f}")
    print(f"Subperiod verdict:      {fv['subperiod_verdict']}")
    print(f"Bootstrap survival:     {fv['bootstrap_survival_rate']:.1%}")
    print(f"Tech concentration:     {fv['tech_concentration']}")
    print(f"Production candidate:   {fv['production_candidate']}")
    print(f"Likely overfit:         {fv['likely_overfit']}")
    print(fv.get("narrative", ""))

    print("\n=== Factor neutralization ===")
    fn = factor_neutral.summary
    print(fn.get("question", ""))
    print(f"Alpha net (market):        {fn['alpha_net_market']:+.2%}")
    print(f"Alpha net (market+sector): {fn['alpha_net_market_sector']:+.2%}")
    print(f"Residual Sharpe (sector):  {fn['residual_sharpe_sector_neutral']:.3f}")
    print(f"Residual DD (sector):      {fn['residual_drawdown_sector_neutral_pct']:.2f}%")
    print(f"Own model verdict:         {fn['verdict_own_model']}")
    print(f"Structure-driven verdict:  {fn['verdict_structure_driven']}")

    print("\n=== Edge decomposition ===")
    bd = edge.beta_decomposition
    print(f"Alpha (ann.):   {bd['alpha_annualized']:+.2%}")
    print(f"Beta vs SPY:    {bd['beta']:.3f}")
    print(f"R-squared:      {bd['r_squared']:.3f}")
    print(f"Edge verdict:   {edge.edge_stability['edge_verdict']}")
    print(edge.narrative.get("question", ""))
    print(f"  Material alpha: {edge.narrative.get('has_material_alpha')}")
    print(f"  Rotation edge:  {edge.narrative.get('rotation_adds_vs_absolute_momentum')}")
    print("\n--- Model breakdown ---")
    print(edge.model_breakdown.to_string(index=False))

    print("\n=== Benchmark vs SPY (analysis) ===")
    s = robustness.summary
    print(f"Excess return vs SPY:  {s['alpha_vs_spy_excess_return']:+.2%}")
    print(f"Full-sample beta:     {s['beta_full_sample']:.3f}")
    print(f"50/50 rank stability: {s['rank_stability_50_50']:.4f}")
    print(f"Momentum-beta proxy:  {s['likely_momentum_beta']}")
    hi = robustness.regime_performance["high_volatility"]
    lo = robustness.regime_performance["low_volatility"]
    print(f"Sharpe high-vol:      {hi['sharpe']:.4f}  (n={hi['n_days']})")
    print(f"Sharpe low-vol:       {lo['sharpe']:.4f}  (n={lo['n_days']})")

    print("\n=== Portfolio results ===")
    print(f"Final capital:  {bt['final_capital']:,.2f}")
    print(f"Return:         {bt['return_pct']:+.2f}%")
    print(f"Max drawdown:   {bt['max_drawdown']:.2f}%")
    print(f"Sharpe:         {bt['sharpe']:.4f}")
    print(f"Trades:         {bt['trades']}")
    print(f"Deflated Sharpe:{validation['deflated_sharpe'].get('dsr', 0):.4f}")
    print(f"Rank stability: {validation['ranking_stability'].get('mean_rank_corr', float('nan')):.4f}")
    print(f"\nArtifacts -> {RESULTS_DIR.resolve()}/")
    print("  multi_asset_equity.csv")
    print("  asset_contributions.csv")
    print("  turnover_report.json")
    print("  benchmark_comparison.csv")
    print("  benchmark_vs_strategy.csv")
    print("  risk_decomposition.json")
    print("  regime_performance.json")
    print("  edge_decomposition.json")
    print("  pnl_attribution.csv")
    print("  model_breakdown.csv")
    print("  factor_neutralization.json")
    print("  factor_neutral_returns.csv")
    print("  robustness_summary.json")
    print("  parameter_surface.csv")
    print("  subperiod_metrics.csv")
    print("  bootstrap_distribution.csv")

    return {
        "backtest": bt,
        "weights": weights,
        "signals": signals,
        "validation": validation,
        "features": feature_panels,
        "robustness": robustness,
        "edge": edge,
        "factor_neutral": factor_neutral,
        "fragility": fragility_report,
    }


if __name__ == "__main__":
    run_multi_asset_pipeline()
