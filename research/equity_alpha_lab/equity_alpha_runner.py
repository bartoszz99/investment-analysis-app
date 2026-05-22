"""
Single-stock cross-sectional alpha lab runner.

Run: python -m research.equity_alpha_lab.equity_alpha_runner

Does NOT modify ETF production stack or add framework layers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

from research.equity_alpha_lab.cross_sectional_features import build_all_features
from research.equity_alpha_lab.forward_returns import build_forward_returns_wide
from research.equity_alpha_lab.hypothesis_tests import run_all_feature_tests
from research.equity_alpha_lab.regime_tests import build_regime_report
from research.equity_alpha_lab.signal_pruning import build_final_report, prune_and_classify
from research.equity_alpha_lab.universe import load_universe

RESULTS = Path("results")
PERIOD = "3y"


def _log(msg: str) -> None:
    print(msg, flush=True)
    sys.stdout.flush()


def run_equity_alpha_lab(*, period: str = PERIOD) -> dict:
    _log("\n=== Equity Alpha Lab (S&P 500 cross-sectional) ===")
    _log("Goal: falsification — not Sharpe optimization")

    panel = load_universe(period=period)
    _log(f"  {len(panel.close.columns)} liquid names | {len(panel.calendar)} days")
    _log(f"  WARNING: {panel.spec.survivorship_warning}")

    features, meta = build_all_features(panel.close, panel.ohlcv)
    _log(f"  Built {len(features)} features across A/B/C")

    ic_rows, neu_rows = run_all_feature_tests(features, panel, meta=meta)

    spy = yf.Ticker("SPY").history(period=period, auto_adjust=True)
    if spy.index.tz is not None:
        spy.index = spy.index.tz_localize(None)
    spy_close = spy["Close"].reindex(panel.calendar).ffill()
    regime_df = build_regime_report(features, build_forward_returns_wide(panel.close)[5], spy_close)

    ic_df = pd.DataFrame(ic_rows)
    neu_df = pd.DataFrame(neu_rows)
    pruning_df, signal_reports = prune_and_classify(neu_df, ic_df)
    final = build_final_report(
        panel.spec.survivorship_warning,
        pruning_df,
        signal_reports,
        len(panel.close.columns),
    )

    RESULTS.mkdir(parents=True, exist_ok=True)
    ic_df.to_csv(RESULTS / "equity_alpha_ic.csv", index=False)
    neu_df.to_csv(RESULTS / "equity_alpha_neutralization.csv", index=False)
    regime_df.to_csv(RESULTS / "equity_alpha_regimes.csv", index=False)
    pruning_df.to_csv(RESULTS / "equity_alpha_pruning.csv", index=False)

    summary = {
        "universe": "sp500",
        "n_tickers": len(panel.close.columns),
        "n_features": len(features),
        "period": period,
        "dropped": panel.dropped,
        "survivorship_warning": panel.spec.survivorship_warning,
        "verdict": final["verdict"],
        "classification_counts": final["classification_counts"],
    }
    with open(RESULTS / "equity_alpha_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    with open(RESULTS / "equity_alpha_final_report.json", "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, default=str)

    _log("\n--- Verdict ---")
    _log(final["verdict"])
    _log("\nArtifacts in results/:")
    for p in (
        "equity_alpha_ic.csv",
        "equity_alpha_summary.json",
        "equity_alpha_regimes.csv",
        "equity_alpha_neutralization.csv",
        "equity_alpha_pruning.csv",
        "equity_alpha_final_report.json",
    ):
        _log(f"  {RESULTS / p}")

    return {"summary": summary, "final": final}


if __name__ == "__main__":
    run_equity_alpha_lab()
