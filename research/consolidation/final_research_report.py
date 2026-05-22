"""
Final research report — consolidation & hypothesis pruning verdict.
Reads existing results only; no new backtests or features.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from research.consolidation.data_loader import load_all_artifacts
from research.consolidation.hypothesis_synthesis import SignalClass, synthesize_all_hypotheses
from research.consolidation.signal_pruning import prune_signals, surviving_hypotheses
from research.consolidation.structural_edge_audit import decompose_total_return

RESULTS = Path("results")


def build_final_verdict(artifacts: dict, pruned: pd.DataFrame, decomposition: pd.DataFrame) -> dict:
    fn = artifacts.get("factor_neutral", {}).get("summary", {})
    rob = artifacts.get("robustness", {})
    alpha_sector = fn.get("alpha_net_market_sector", 0) or 0
    alpha_market = fn.get("alpha_net_market", 0) or 0
    res_sharpe_sector = fn.get("residual_sharpe_sector_neutral", 0) or 0

    survivors = surviving_hypotheses(pruned)
    true_edge_count = len(survivors[survivors["classification"] == SignalClass.TRUE_EDGE.value])

    true_alpha_exists = alpha_sector > 0.02 and res_sharpe_sector > 0.3 and true_edge_count > 0

    # Dominant driver
    if rob.get("tech_concentration") and alpha_sector <= 0:
        dominant = "sector"
    elif fn.get("mean_momentum_beta_20d", 0) > 0.5:
        dominant = "momentum"
    elif alpha_market > alpha_sector:
        dominant = "beta"
    else:
        dominant = "residual" if alpha_sector > 0 else "sector"

    fragility = rob.get("fragility_score", 1.0) or 1.0
    bootstrap = rob.get("bootstrap_survival_rate", 0) or 0
    overfit = "high" if rob.get("likely_overfit") else ("medium" if fragility > 0.5 else "low")

    if true_alpha_exists:
        system_type = "alpha system"
        investable = "yes"
        confidence = 65
    elif alpha_market > 0 and alpha_sector <= 0:
        system_type = "overlay system"
        investable = "no"
        confidence = 75
    else:
        system_type = "noise system"
        investable = "no"
        confidence = 80

    # Research hypotheses worth further study (not production)
    research_candidates = []
    pf = artifacts.get("passive_flow_summary", {})
    q1 = pf.get("Q1_concentration_predicts_fragility", {})
    if q1.get("structural_plausibility"):
        research_candidates.append(
            {
                "hypothesis": "passive_flow_narrow_rally_tail_asymmetry",
                "type": "STRUCTURAL_RESEARCH",
                "note": "Negative skew after narrow rallies — not tradable alpha, worth structural study",
            }
        )

    return {
        "true_alpha_exists": "YES" if true_alpha_exists else "NO",
        "dominant_driver": dominant,
        "system_type": system_type,
        "overfit_risk": overfit,
        "investable": investable,
        "confidence": confidence,
        "metrics": {
            "alpha_market_annual": alpha_market,
            "alpha_sector_neutral_annual": alpha_sector,
            "residual_sharpe_sector_neutral": res_sharpe_sector,
            "bootstrap_survival_rate": bootstrap,
            "fragility_score": fragility,
            "tech_concentration": rob.get("tech_concentration"),
            "production_candidate": rob.get("production_candidate"),
        },
        "surviving_signal_count": len(survivors),
        "true_edge_survivors": true_edge_count,
        "research_hypotheses_for_further_study": research_candidates,
        "honest_conclusion": _honest_conclusion(true_alpha_exists, dominant, alpha_sector, rob),
    }


def _honest_conclusion(
    true_alpha: bool,
    dominant: str,
    alpha_sector: float,
    rob: dict,
) -> str:
    if true_alpha:
        return "Marginal independent alpha detected — requires OOS confirmation before any production use."
    parts = [
        "NO independent residual alpha after sector + momentum neutralization.",
        f"Dominant return driver: {dominant} exposure (not a standalone edge).",
        f"Sector-neutral alpha: {alpha_sector:.1%}.",
    ]
    if rob.get("tech_concentration"):
        parts.append("Performance is effectively a QQQ/XLK momentum overlay during favorable bull regime.")
    parts.append("Do NOT optimize further — pivot to structural market research (breadth/passive flow tails).")
    return " ".join(parts)


def run_consolidation() -> dict:
    print("\n=== CONSOLIDATION & HYPOTHESIS PRUNING ===")
    print("Reading existing results only — no new pipelines.\n")

    artifacts = load_all_artifacts()
    missing = [k for k, v in artifacts.items() if (isinstance(v, dict) and not v) or (isinstance(v, pd.DataFrame) and v.empty)]
    if missing:
        print(f"  Warning: missing/empty artifacts: {missing[:8]}")

    synthesis = synthesize_all_hypotheses(artifacts)
    pruned = prune_signals(synthesis, artifacts)
    decomposition = decompose_total_return(artifacts)
    verdict = build_final_verdict(artifacts, pruned, decomposition)

    RESULTS.mkdir(parents=True, exist_ok=True)
    pruned.to_csv(RESULTS / "hypothesis_pruning.csv", index=False)
    decomposition.to_csv(RESULTS / "edge_decomposition_final.csv", index=False)

    report = {
        "verdict": verdict,
        "classification_summary": synthesis["classification"].value_counts().to_dict()
        if not synthesis.empty
        else {},
        "pruning_summary": {
            "total_signals": len(pruned),
            "pruned": int(pruned["pruned"].sum()),
            "kept": int(pruned["kept"].sum()),
        },
        "surviving_hypotheses": surviving_hypotheses(pruned).to_dict(orient="records"),
        "edge_decomposition": decomposition.to_dict(orient="records"),
        "methodology": (
            "Consolidation reads prior research outputs. "
            "Pruning: neutral IC<0.05, HIGH leakage, sector-alpha<=0, production_candidate=false. "
            "No optimization performed."
        ),
    }
    with open(RESULTS / "final_research_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    print("--- FINAL VERDICT ---")
    print(f"  True alpha exists?  {verdict['true_alpha_exists']}")
    print(f"  Dominant driver:    {verdict['dominant_driver']}")
    print(f"  System type:        {verdict['system_type']}")
    print(f"  Investable?         {verdict['investable']}")
    print(f"  Confidence:         {verdict['confidence']}/100")
    print(f"  Overfit risk:       {verdict['overfit_risk']}")
    print(f"\n  {verdict['honest_conclusion']}")
    print(f"\n  Saved: {RESULTS / 'final_research_report.json'}")
    print(f"         {RESULTS / 'hypothesis_pruning.csv'}")
    print(f"         {RESULTS / 'edge_decomposition_final.csv'}")

    return report


if __name__ == "__main__":
    run_consolidation()
