"""
Signal pruning — logical elimination of non-viable hypotheses.
Does not delete code; marks signals as PRUNED with reason.
"""

from __future__ import annotations

import pandas as pd

from research.consolidation.hypothesis_synthesis import SignalClass

IC_THRESHOLD = 0.05
BOOTSTRAP_SURVIVAL_MIN = 0.45
REGIME_FAIL_COUNT = 2


def _regime_stability(signal: str, regime_df: pd.DataFrame) -> tuple[bool, int]:
    if regime_df.empty or "event" not in regime_df.columns:
        return True, 0
    # For breadth/alternative — use passive flow regimes as proxy for cross-regime IC sign stability
    sub = regime_df[regime_df.get("event", pd.Series()).astype(str).str.contains("narrow", na=False)]
    if sub.empty:
        return True, 0
    signs = sub["mean_diff"].dropna()
    if len(signs) < 2:
        return True, 0
    # inconsistent = mixed positive/negative mean_diff across calendar regimes
    pos = (signs > 0).sum()
    neg = (signs < 0).sum()
    inconsistent = min(pos, neg) >= REGIME_FAIL_COUNT
    return not inconsistent, int(min(pos, neg))


def prune_signals(
    synthesis_df: pd.DataFrame,
    artifacts: dict,
) -> pd.DataFrame:
    rob = artifacts.get("robustness", {})
    fn = artifacts.get("factor_neutral", {}).get("summary", {})
    bootstrap_survival = rob.get("bootstrap_survival_rate", 1.0) or 1.0
    regime_df = artifacts.get("passive_flow_regimes", pd.DataFrame())

    rows = []
    for _, r in synthesis_df.iterrows():
        pruned = False
        reasons = []

        ic_n = r.get("ic_neutral_20d")
        if ic_n is not None and pd.notna(ic_n) and abs(ic_n) < IC_THRESHOLD:
            if r["classification"] != SignalClass.STRUCTURAL_EXPOSURE.value:
                pruned = True
                reasons.append(f"|neutral IC| < {IC_THRESHOLD}")

        if str(r.get("leakage_risk", "")).upper() == "HIGH":
            pruned = True
            reasons.append("HIGH leakage risk")

        if r["signal"] == "momentum_etf_rotation_system":
            alpha_sector = fn.get("alpha_net_market_sector", 0) or 0
            if alpha_sector <= 0:
                pruned = True
                reasons.append(f"sector-neutral alpha {alpha_sector:.1%} <= 0")
            if not rob.get("production_candidate", False):
                pruned = True
                reasons.append("robustness: production_candidate=false")
            if bootstrap_survival < BOOTSTRAP_SURVIVAL_MIN:
                pruned = True
                reasons.append(f"bootstrap survival {bootstrap_survival:.0%} < {BOOTSTRAP_SURVIVAL_MIN:.0%}")

        stable, fail_n = _regime_stability(r["signal"], regime_df)
        if not stable and r["classification"] == SignalClass.TRUE_EDGE.value:
            pruned = True
            reasons.append(f"unstable in {fail_n}+ regimes")

        if r["classification"] == SignalClass.NOISE.value:
            pruned = True
            reasons.append("classified NOISE")

        kept = not pruned
        rows.append(
            {
                **r.to_dict(),
                "pruned": pruned,
                "kept": kept,
                "prune_reasons": "; ".join(reasons) if reasons else "",
            }
        )

    return pd.DataFrame(rows)


def surviving_hypotheses(pruned_df: pd.DataFrame) -> pd.DataFrame:
    return pruned_df[pruned_df["kept"] == True].copy()  # noqa: E712
