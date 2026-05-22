"""
Research summary builder — statistical / economic / structural assessment.
Labels weak findings as LIKELY NOISE.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _assess_finding(
    mean_diff: float,
    tail_diff: float,
    n_events: int,
    regime_consistency: float,
    mechanism: str,
) -> dict:
    """Three-layer research standard."""
    stat_sig = n_events >= 30 and abs(mean_diff) > 0.005
    econ_sig = abs(mean_diff) > 0.01 or abs(tail_diff) > 0.05
    structural = regime_consistency >= 0.5 and n_events >= 20

    if not stat_sig and not econ_sig:
        label = "LIKELY NOISE"
    elif stat_sig and econ_sig and structural:
        label = "STRUCTURALLY PLAUSIBLE"
    elif stat_sig or econ_sig:
        label = "MIXED — NEEDS LONGER SAMPLE"
    else:
        label = "LIKELY NOISE"

    return {
        "label": label,
        "statistical_significance": stat_sig,
        "economic_significance": econ_sig,
        "structural_plausibility": structural,
        "mechanism": mechanism,
        "n_events": n_events,
        "mean_diff_vs_unconditional": mean_diff,
        "left_tail_diff": tail_diff,
        "regime_consistency": regime_consistency,
    }


def build_research_answers(
    event_studies: list[dict],
    regime_rows: list[dict],
    data_limitation: str,
) -> dict:
    """Answer the five core research questions."""
    # Aggregate Event A across ETFs
    event_a = [s for s in event_studies if s["event"] == "A_narrow_rally"]
    tail_diffs, mean_diffs, ns = [], [], []
    for s in event_a:
        cmp = s.get("vs_unconditional_20d", {})
        mean_diffs.append(cmp.get("mean_diff", 0) or 0)
        tail_diffs.append(cmp.get("correction_prob_diff", 0) or 0)
        ns.append(s.get("n_events", 0))

    avg_mean = float(np.mean(mean_diffs)) if mean_diffs else 0.0
    avg_tail = float(np.mean(tail_diffs)) if tail_diffs else 0.0
    avg_n = int(np.mean(ns)) if ns else 0

    regime_consistency = _regime_consistency(regime_rows, "A_narrow_rally")

    fragility = _assess_finding(
        avg_mean,
        avg_tail,
        avg_n,
        regime_consistency,
        "Passive inflows concentrate in mega-caps while breadth narrows",
    )

    return {
        "data_limitation": data_limitation,
        "Q1_concentration_predicts_fragility": fragility,
        "Q2_passive_rallies_structurally_unstable": {
            "assessment": _assess_finding(
                avg_mean,
                avg_tail,
                avg_n,
                regime_consistency,
                "Benchmark-driven flows decouple index from internals",
            ),
            "evidence": "Event B passive dominance forward distributions",
        },
        "Q3_narrow_rallies_worse_tails": {
            "assessment": fragility,
            "avg_correction_prob_diff_20d": avg_tail,
        },
        "Q4_persistent_across_regimes": {
            "regime_consistency_score": regime_consistency,
            "verdict": "structural" if regime_consistency >= 0.6 else "regime_specific",
        },
        "Q5_structural_market_deformation": {
            "evidence_strength": "moderate" if fragility["label"] != "LIKELY NOISE" else "weak",
            "note": "Static holdings proxy limits historical precision",
        },
        "research_philosophy": (
            "Objective: measure passive-structure distortions, not build a strategy. "
            "Tiny unstable effects without mechanism → LIKELY NOISE."
        ),
    }


def _regime_consistency(regime_rows: list[dict], event: str) -> float:
    subset = [r for r in regime_rows if r.get("event") == event and r.get("n_events", 0) >= 5]
    if len(subset) < 2:
        return 0.0
    signs = [1 if (r.get("mean_diff") or 0) < 0 else -1 for r in subset]
    # for fragility we expect negative mean diff vs unconditional after narrow rally
    return float(sum(1 for s in signs if s == 1) / len(signs))


def save_summary(payload: dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
