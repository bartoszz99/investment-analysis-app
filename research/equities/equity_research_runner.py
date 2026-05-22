"""
Cross-sectional equity research runner — isolated branch.
Run: python -m research.equities.equity_research_runner

Does NOT modify main.py, multi_asset_runner.py, or ETF framework.
"""

from __future__ import annotations

import json
from pathlib import Path

from research.equities.data_pipeline import load_equity_panel, load_spy_benchmark
from research.equities.evaluation import write_results
from research.equities.hypothesis_tests import run_all_hypotheses
from research.equities.universe import get_universe

RESULTS = Path("results")
PERIOD = "3y"
UNIVERSE_NAME = "nasdaq100"
MAX_SIZE = 100


def run_equity_research(
    *,
    period: str = PERIOD,
    universe_name: str = UNIVERSE_NAME,
    max_size: int = MAX_SIZE,
) -> dict:
    import sys

    def _log(msg: str) -> None:
        print(msg, flush=True)
        sys.stdout.flush()

    _log("\n=== Cross-Sectional Equity Research Lab ===")
    _log("Phase: signal research only (no portfolio optimization)")
    _log(f"Universe: liquid large caps — {universe_name} (max {max_size})")

    spec = get_universe(universe_name, max_size=max_size)  # type: ignore[arg-type]
    panel = load_equity_panel(spec, period=period)
    spy_close = load_spy_benchmark(period).reindex(panel.calendar).ffill()

    _log(f"  Loaded {len(panel.close.columns)} names, {len(panel.calendar)} days")
    if panel.dropped:
        _log(f"  Dropped illiquid: {len(panel.dropped)} tickers")
    _log("  Running hypotheses A -> C ...")

    out = run_all_hypotheses(
        panel.ohlcv,
        panel.close,
        spy_close,
        tuple(panel.close.columns),
    )

    summary = {
        "universe": spec.name,
        "n_tickers": len(panel.close.columns),
        "period": period,
        "n_days": len(panel.calendar),
        "dropped_tickers": panel.dropped,
        "falsification": out["hypothesis_report"].get("falsification", {}),
        "anti_leakage": [
            "rolling features use shift(1)",
            "forward returns future-only",
            "sector neutralization cross-sectional same-day demean",
            "beta/momentum neutralization trailing OLS only",
            "no global z-score or full-sample normalization",
        ],
    }

    write_results(
        out["ic_rows"],
        out["neutral_rows"],
        out["regime_rows"],
        summary,
        out["hypothesis_report"],
        results_dir=RESULTS,
    )

    _log("\n--- Falsification verdict ---")
    _log(json.dumps(summary["falsification"], indent=2))
    _log("\nArtifacts:")
    for name in (
        "equity_signal_ic.csv",
        "equity_signal_summary.json",
        "equity_neutralization.csv",
        "equity_regime_analysis.csv",
        "equity_hypothesis_report.json",
    ):
        _log(f"  {RESULTS / name}")

    return {"summary": summary, **out}


if __name__ == "__main__":
    run_equity_research()
