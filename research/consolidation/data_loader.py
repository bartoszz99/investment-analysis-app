"""
Load existing research artifacts — no new computation pipelines.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RESULTS = Path("results")


def _load_json(name: str) -> dict:
    p = RESULTS / name
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def _load_csv(name: str) -> pd.DataFrame:
    p = RESULTS / name
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def load_all_artifacts() -> dict:
    """Aggregate every results file used in consolidation."""
    return {
        "robustness": _load_json("robustness_summary.json"),
        "factor_neutral": _load_json("factor_neutralization.json"),
        "edge_decomposition": _load_json("edge_decomposition.json"),
        "breadth_summary": _load_json("breadth_summary.json"),
        "liquidity_summary": _load_json("liquidity_summary.json"),
        "passive_flow_summary": _load_json("passive_flow_summary.json"),
        "alternative_tests": _load_json("alternative_signal_tests.json"),
        "regime_performance": _load_json("regime_performance.json"),
        "benchmark_vs_strategy": _load_csv("benchmark_vs_strategy.csv"),
        "model_breakdown": _load_csv("model_breakdown.csv"),
        "pnl_attribution": _load_csv("pnl_attribution.csv"),
        "ic_analysis": _load_csv("ic_analysis.csv"),
        "breadth_ic": _load_csv("breadth_ic.csv"),
        "liquidity_ic": _load_csv("liquidity_ic.csv"),
        "passive_flow_events": _load_csv("passive_flow_events.csv"),
        "passive_flow_regimes": _load_csv("passive_flow_regimes.csv"),
        "parameter_surface": _load_csv("parameter_surface.csv"),
    }
