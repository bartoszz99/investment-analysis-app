"""
Classify each signal — falsification-first labels.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd

IC_MIN = 0.02
IC_NEUTRAL_MIN = 0.015
BOOTSTRAP_MIN = 0.45
OOS_RATIO_MIN = 0.5
COST_SPREAD_MIN = 0.0005


class SignalClass(str, Enum):
    TRUE_EDGE = "TRUE_EDGE"
    STRUCTURAL_EXPOSURE = "STRUCTURAL_EXPOSURE"
    REGIME_DEPENDENT = "REGIME_DEPENDENT"
    LIKELY_NOISE = "LIKELY_NOISE"
    HIGH_LEAKAGE_RISK = "HIGH_LEAKAGE_RISK"


def _collapse_ratio(raw: float, neutral: float) -> float:
    if raw is None or neutral is None or np.isnan(raw) or abs(raw) < 1e-6:
        return 1.0
    return abs(neutral) / abs(raw)


def classify_signal(row: dict, ic_row: dict | None = None) -> tuple[SignalClass, list[str]]:
    reasons: list[str] = []
    leakage = str(row.get("leakage_risk", "")).upper()
    if "HIGH" in leakage or "LEAKAGE" in leakage:
        return SignalClass.HIGH_LEAKAGE_RISK, ["earnings/calendar or proxy timestamps unreliable"]

    raw = row.get("ic_raw_5d", np.nan)
    sec = row.get("ic_sector_5d", np.nan)
    beta = row.get("ic_beta_5d", np.nan)
    mom = row.get("ic_momentum_5d", np.nan)

    if raw is None or (raw != raw) or abs(raw) < IC_MIN:
        return SignalClass.LIKELY_NOISE, [f"|raw IC| < {IC_MIN}"]

    if _collapse_ratio(raw, sec) < 0.5 or _collapse_ratio(raw, beta) < 0.5:
        reasons.append("IC collapses after sector/beta neutralization")
        return SignalClass.STRUCTURAL_EXPOSURE, reasons

    if mom == mom and _collapse_ratio(raw, mom) < 0.5:
        reasons.append("IC collapses after momentum neutralization")
        return SignalClass.STRUCTURAL_EXPOSURE, reasons

    if ic_row:
        ic_on = ic_row.get("ic_regime_on", np.nan)
        ic_off = ic_row.get("ic_regime_off", np.nan)
        if ic_on == ic_on and ic_off == ic_off and ic_on * ic_off < 0 and abs(ic_on) > IC_MIN and abs(ic_off) > IC_MIN:
            return SignalClass.REGIME_DEPENDENT, ["opposite IC sign bull vs off-regime"]

        oos = ic_row.get("ic_out_of_sample", np.nan)
        ins = ic_row.get("ic_in_sample", np.nan)
        if ins == ins and oos == oos and abs(ins) > IC_MIN:
            if abs(oos) < abs(ins) * OOS_RATIO_MIN:
                return SignalClass.REGIME_DEPENDENT, ["OOS IC weak vs in-sample"]

        boot = ic_row.get("bootstrap_survival", np.nan)
        if boot == boot and boot < BOOTSTRAP_MIN:
            return SignalClass.LIKELY_NOISE, [f"bootstrap survival {boot:.0%} < {BOOTSTRAP_MIN:.0%}"]

        spread_c = ic_row.get("quintile_spread_after_cost", np.nan)
        if spread_c == spread_c and abs(spread_c) < COST_SPREAD_MIN:
            return SignalClass.LIKELY_NOISE, ["quintile spread does not survive cost haircut"]

    best_neutral = max(
        [abs(x) for x in (sec, beta, mom) if x == x],
        default=0,
    )
    if best_neutral >= IC_NEUTRAL_MIN:
        return SignalClass.TRUE_EDGE, ["neutralized IC remains above threshold — requires manual mechanism review"]

    return SignalClass.LIKELY_NOISE, ["weak IC after all neutralizations"]


def prune_and_classify(
    neu_df: pd.DataFrame,
    ic_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict]]:
    raw_ic = ic_df[
        (ic_df["horizon"] == 5)
        & (ic_df["neutralization"].isin(["raw", "market_ts"]))
    ]
    ic_lookup: dict = {}
    for _, r in raw_ic.iterrows():
        if r["feature"] not in ic_lookup:
            ic_lookup[r["feature"]] = r.to_dict()

    rows = []
    reports = []
    for _, r in neu_df.drop_duplicates("feature").iterrows():
        feat = r["feature"]
        ic_row = ic_lookup.get(feat, {})
        cls, reasons = classify_signal(r.to_dict(), ic_row)
        kept = cls == SignalClass.TRUE_EDGE
        rows.append(
            {
                "feature": feat,
                "category": r.get("category"),
                "classification": cls.value,
                "kept": kept,
                "reasons": "; ".join(reasons),
                "ic_raw_5d": r.get("ic_raw_5d"),
                "ic_sector_5d": r.get("ic_sector_5d"),
                "ic_beta_5d": r.get("ic_beta_5d"),
                "ic_momentum_5d": r.get("ic_momentum_5d"),
            }
        )
        reports.append(_final_report_entry(feat, cls, reasons, r, ic_row))

    return pd.DataFrame(rows), reports


def _final_report_entry(feat, cls, reasons, neu_row, ic_row) -> dict:
    raw = neu_row.get("ic_raw_5d", np.nan)
    sec = neu_row.get("ic_sector_5d", np.nan)
    return {
        "feature": feat,
        "classification": cls.value,
        "reasons": reasons,
        "questions": {
            "independent_alpha_survives": cls == SignalClass.TRUE_EDGE,
            "economically_meaningful": raw == raw and abs(raw) > IC_MIN,
            "stable_across_regimes": cls not in (
                SignalClass.REGIME_DEPENDENT,
                SignalClass.LIKELY_NOISE,
            ),
            "tail_concentrated": ic_row.get("spread_tail_high_signal_days") if ic_row else None,
            "survives_costs": ic_row.get("quintile_spread_after_cost") if ic_row else None,
            "mechanism_plausible": cls not in (
                SignalClass.LIKELY_NOISE,
                SignalClass.HIGH_LEAKAGE_RISK,
            ),
        },
    }


def build_final_report(
    spec_warning: str,
    pruning_df: pd.DataFrame,
    signal_reports: list[dict],
    n_tickers: int,
) -> dict:
    counts = pruning_df["classification"].value_counts().to_dict()
    true_edges = pruning_df[pruning_df["classification"] == SignalClass.TRUE_EDGE.value]["feature"].tolist()
    return {
        "universe": "sp500",
        "n_tickers": n_tickers,
        "survivorship_warning": spec_warning,
        "classification_counts": counts,
        "true_edge_candidates": true_edges,
        "verdict": (
            "NO structurally independent cross-sectional alpha confirmed"
            if not true_edges
            else f"{len(true_edges)} candidate(s) need mechanism review — not production"
        ),
        "signals": signal_reports,
        "philosophy": "Falsification over Sharpe optimization.",
    }
