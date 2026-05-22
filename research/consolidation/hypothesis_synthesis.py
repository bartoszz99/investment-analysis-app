"""
Hypothesis synthesis — classify every signal into:
  TRUE_EDGE | STRUCTURAL_EXPOSURE | NOISE
"""

from __future__ import annotations

from enum import Enum

import pandas as pd

IC_NEUTRAL_THRESHOLD = 0.05


class SignalClass(str, Enum):
    TRUE_EDGE = "TRUE_EDGE"
    STRUCTURAL_EXPOSURE = "STRUCTURAL_EXPOSURE"
    NOISE = "NOISE"


def _classify_ic_row(row: pd.Series, source: str) -> dict:
    ic_n = row.get("ic_20d_neutral") or row.get("ic_neutral")
    ic_raw = row.get("ic_20d") or row.get("ic_spearman")
    leakage = str(row.get("leakage_risk", "LOW")).upper()

    if pd.isna(ic_n):
        ic_n = 0.0
    if pd.isna(ic_raw):
        ic_raw = 0.0

    # Hard reject: leakage risk HIGH → never TRUE_EDGE
    if leakage == "HIGH":
        cls = SignalClass.NOISE
        reason = "HIGH leakage risk — cannot trust timing"
    elif abs(ic_n) >= IC_NEUTRAL_THRESHOLD:
        cls = SignalClass.TRUE_EDGE
        reason = f"Neutral IC {ic_n:.3f} >= threshold (needs regime/bootstrap confirm)"
    elif abs(ic_raw) >= 0.08 and abs(ic_n) < IC_NEUTRAL_THRESHOLD:
        cls = SignalClass.STRUCTURAL_EXPOSURE
        reason = f"Raw IC {ic_raw:.3f} collapses after SPY neutral ({ic_n:.3f}) — beta/sector proxy"
    else:
        cls = SignalClass.NOISE
        reason = f"Neutral IC {ic_n:.3f} below threshold — likely noise"

    return {
        "signal": row.get("signal", "unknown"),
        "target": row.get("target", "unknown"),
        "source_module": source,
        "classification": cls.value,
        "ic_raw_20d": float(ic_raw) if ic_raw == ic_raw else None,
        "ic_neutral_20d": float(ic_n) if ic_n == ic_n else None,
        "leakage_risk": leakage,
        "reason": reason,
    }


def synthesize_alternative_signals(ic_df: pd.DataFrame) -> list[dict]:
    if ic_df.empty:
        return []
    # one row per signal (best target by |neutral ic|)
    ic_col = "ic_20d_neutral_spy" if "ic_20d_neutral_spy" in ic_df.columns else "ic_20d_neutral"
    if ic_col not in ic_df.columns:
        ic_df = ic_df.copy()
        ic_df["ic_20d_neutral_spy"] = ic_df.get("ic_neutral")
        ic_col = "ic_20d_neutral_spy"
    rows = []
    for signal, grp in ic_df.groupby("signal"):
        if ic_col in grp.columns and grp[ic_col].notna().any():
            best = grp.loc[grp[ic_col].abs().idxmax()]
        else:
            best = grp.iloc[0]
        best = best.copy()
        best["signal"] = signal
        best["ic_20d_neutral"] = best.get(ic_col)
        best["ic_20d"] = best.get("ic_20d", best.get("ic_20d_neutral"))
        rows.append(_classify_ic_row(best, "alternative_data"))
    return rows


def synthesize_breadth_signals(ic_df: pd.DataFrame) -> list[dict]:
    if ic_df.empty:
        return []
    ic_df = ic_df[ic_df["horizon"] == 20] if "horizon" in ic_df.columns else ic_df
    rows = []
    for signal, grp in ic_df.groupby("signal"):
        if "ic_neutral" in grp.columns and grp["ic_neutral"].notna().any():
            best = grp.loc[grp["ic_neutral"].abs().idxmax()]
        else:
            best = grp.iloc[0]
        rows.append(_classify_ic_row(best, "breadth"))
    return rows


def synthesize_liquidity_signals(ic_df: pd.DataFrame) -> list[dict]:
    if ic_df.empty:
        return []
    ic_df = ic_df[ic_df["horizon"] == 20] if "horizon" in ic_df.columns else ic_df
    rows = []
    for signal, grp in ic_df.groupby("signal"):
        if "ic_neutral" in grp.columns and grp["ic_neutral"].notna().any():
            best = grp.loc[grp["ic_neutral"].abs().idxmax()]
        else:
            best = grp.iloc[0]
        rows.append(_classify_ic_row(best, "liquidity"))
    return rows


def synthesize_production_system(artifacts: dict) -> list[dict]:
    """Classify the momentum ETF allocator as a system-level hypothesis."""
    fn = artifacts.get("factor_neutral", {}).get("summary", {})
    rob = artifacts.get("robustness", {})
    alpha_sector = fn.get("alpha_net_market_sector", 0) or 0
    alpha_market = fn.get("alpha_net_market", 0) or 0
    res_sharpe_sector = fn.get("residual_sharpe_sector_neutral", 0) or 0

    if alpha_sector > 0.02 and res_sharpe_sector > 0.3:
        cls = SignalClass.TRUE_EDGE
        reason = "Positive sector-neutral alpha with residual Sharpe"
    elif alpha_market > 0.02 and alpha_sector <= 0:
        cls = SignalClass.STRUCTURAL_EXPOSURE
        reason = (
            f"Market alpha {alpha_market:.1%} but sector-neutral alpha {alpha_sector:.1%} — "
            "sector/momentum tilt not independent edge"
        )
    else:
        cls = SignalClass.STRUCTURAL_EXPOSURE if alpha_market > 0 else SignalClass.NOISE
        reason = f"Sector-neutral alpha {alpha_sector:.1%}, residual Sharpe sector {res_sharpe_sector:.2f}"

    return [
        {
            "signal": "momentum_etf_rotation_system",
            "target": "multi_asset_portfolio",
            "source_module": "multi_asset_runner",
            "classification": cls.value,
            "ic_raw_20d": None,
            "ic_neutral_20d": alpha_sector,
            "leakage_risk": "LOW",
            "reason": reason,
            "tech_concentration": rob.get("tech_concentration"),
            "production_candidate": rob.get("production_candidate"),
        }
    ]


def synthesize_passive_flow(artifacts: dict) -> list[dict]:
    pf = artifacts.get("passive_flow_summary", {})
    q1 = pf.get("Q1_concentration_predicts_fragility", {})
    label = q1.get("label", "UNKNOWN")
    cls = (
        SignalClass.STRUCTURAL_EXPOSURE
        if "MIXED" in label or "STRUCTURALLY" in label
        else SignalClass.NOISE
    )
    return [
        {
            "signal": "passive_flow_narrow_rally_fragility",
            "target": "SPY/QQQ/XLK",
            "source_module": "passive_flow",
            "classification": cls.value,
            "ic_raw_20d": q1.get("mean_diff_vs_unconditional"),
            "ic_neutral_20d": None,
            "leakage_risk": "MEDIUM",
            "reason": f"Event study: {label} — structural tail asymmetry, not tradable alpha",
        }
    ]


def synthesize_all_hypotheses(artifacts: dict) -> pd.DataFrame:
    rows = []
    rows.extend(synthesize_production_system(artifacts))
    rows.extend(synthesize_alternative_signals(artifacts.get("ic_analysis", pd.DataFrame())))
    rows.extend(synthesize_breadth_signals(artifacts.get("breadth_ic", pd.DataFrame())))
    rows.extend(synthesize_liquidity_signals(artifacts.get("liquidity_ic", pd.DataFrame())))
    rows.extend(synthesize_passive_flow(artifacts))
    return pd.DataFrame(rows)
