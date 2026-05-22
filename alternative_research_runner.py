"""
Alternative information research runner.
STRICTLY separate from production: execution, portfolio, benchmark, neutralization unchanged.

Tests whether non-price information has predictive power beyond SPY/sector beta.
"""

from __future__ import annotations

from pathlib import Path

import config
from layers.data_layer import load_multi_asset
from research.alternative_data.earnings_signals import build_earnings_factors
from research.alternative_data.flow_signals import build_flow_signals
from research.alternative_data.macro_signals import build_macro_signals
from research.alternative_data.research_tests import run_all_signal_tests, save_research_outputs
from research.alternative_data.sentiment_signals import build_sentiment_signals
from research.alternative_data.signal_registry import SignalRegistry

RESULTS_DIR = Path("results")


def run_alternative_research(period: str | None = None) -> dict:
    period = period or config.MULTI_ASSET_PERIOD
    tickers = list(config.ETF_UNIVERSE)

    print(f"\n=== Alternative information research ({period}) ===")
    print("Production stack NOT modified — research layer only.\n")

    bundle = load_multi_asset(tickers, period, forward_fill_alignment=False)
    tradable = bundle.drop_non_tradable()
    calendar = tradable.calendar
    close_wide = tradable.wide("Close")
    volume_wide = tradable.wide("Volume")

    registry = SignalRegistry()

    # Build all signal families
    for sig in build_earnings_factors(calendar, ("SPY", "QQQ")).values():
        registry.register(sig)
    for sig in build_macro_signals(calendar, period).values():
        registry.register(sig)
    for sig in build_flow_signals(close_wide, volume_wide).values():
        registry.register(sig)
    for sig in build_sentiment_signals(calendar, period).values():
        registry.register(sig)

    print(f"Registered signals: {len(registry.list_signals())}")
    for row in registry.summary_table()[:8]:
        print(
            f"  {row['name']}: lag={row['lag_days']} risk={row['leakage_risk']} n={row['n_obs']}"
        )
    if len(registry.list_signals()) > 8:
        print(f"  ... +{len(registry.list_signals()) - 8} more")

    all_results, ic_df, regime_df = run_all_signal_tests(registry, close_wide)
    save_research_outputs(all_results, ic_df, regime_df, registry.summary_table(), RESULTS_DIR)
    registry.to_json(RESULTS_DIR / "alternative_signal_registry.json")

    # Highlight candidates
    from research.alternative_data.research_tests import rank_signals_by_neutral_ic

    ranked = rank_signals_by_neutral_ic(ic_df)
    print("\n=== Top signals by |neutral IC| vs SPY ===")
    cols = ["signal", "mean_ic_20d", "mean_ic_20d_neutral", "stable_neutral_ic"]
    print(ranked[cols].head(8).to_string(index=False))

    stable = ranked[ranked["stable_neutral_ic"]]
    print(f"\nStable neutral IC candidates: {len(stable)}")
    if not stable.empty:
        print(stable[["signal", "mean_ic_20d_neutral"]].head(5).to_string(index=False))
    else:
        print("  (none meet threshold |IC|>0.03, std<0.15 — expected for first pass)")

    print(f"\nArtifacts -> {RESULTS_DIR.resolve()}/")
    print("  alternative_signal_tests.json")
    print("  ic_analysis.csv")
    print("  regime_signal_analysis.csv")
    print("  alternative_signal_registry.json")

    return {
        "registry": registry,
        "ic_analysis": ic_df,
        "regime_analysis": regime_df,
        "ranked": ranked,
        "all_results": all_results,
    }


if __name__ == "__main__":
    run_alternative_research()
