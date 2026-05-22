"""
Phase-1 hypothesis tests — A (earnings), B (breadth), C (liquidity).
Falsification-first; documents structural basis and leakage.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from research.equities.evaluation import evaluate_feature_panel
from research.equities.feature_library import (
    build_breadth_features_wide,
    build_earnings_panel,
    liquidity_features_wide,
)
from research.equities.neutralization import (
    apply_neutralization_suite,
    classify_structural_exposure,
)
from research.equities.universe import sector_series

HYPOTHESIS_META: dict[str, dict] = {
    "A_post_earnings_drift": {
        "mechanism": "Institutions rebalance slowly after earnings surprises.",
        "counterparty": "Retail / fast money on headline; institutions on schedule.",
        "forced_flow": "Mandate rebalance, risk limits, quarterly reporting windows.",
        "arb_friction": "Position limits, uncertainty dispersion, staggered disclosure.",
        "cost_survival": "Effect must exceed spread + short horizon drift noise.",
        "structural_basis": "CONDITIONAL — valid only with reliable event timestamps.",
        "leakage_risk": "HIGH — yfinance earnings_dates not exchange-official.",
    },
    "B_sector_breadth_deterioration": {
        "mechanism": "Index/sector rises while fewer stocks participate — fragile rally.",
        "counterparty": "Passive index buyers vs weak single-stock sponsors.",
        "forced_flow": "Index inclusion flows, ETF creation without stock-level demand.",
        "arb_friction": "Sector basket hedging imperfect; breadth not directly tradable.",
        "cost_survival": "Slow-moving; 5–20d horizon may absorb costs if real.",
        "structural_basis": "YES — participation divergence is observable market structure.",
        "leakage_risk": "LOW — features use lagged component prices only.",
    },
    "C_liquidity_exhaustion": {
        "mechanism": "Crowded moves exhaust liquidity; short-term mean reversion.",
        "counterparty": "Late momentum chasers vs liquidity providers.",
        "forced_flow": "Deleveraging, margin calls, vol-targeting reductions.",
        "arb_friction": "Inventory risk, gap risk, asymmetric funding.",
        "cost_survival": "Requires high turnover; 5d horizon sensitive to costs.",
        "structural_basis": "YES — microstructure stress; test if survives neutralization.",
        "leakage_risk": "LOW — standard OHLCV with shift(1).",
    },
}


def _spy_regime_mask(spy_close: pd.Series, ma_window: int = 200) -> pd.Series:
    lag = spy_close.shift(1)
    ma = lag.rolling(ma_window, min_periods=ma_window).mean()
    return (lag > ma).reindex(spy_close.index)


def run_hypothesis_c(
    ohlcv: dict[str, pd.DataFrame],
    close: pd.DataFrame,
    *,
    sectors: dict[str, str],
    spy_ret: pd.Series,
    regime_mask: pd.Series,
) -> tuple[list[dict], list[dict], list[dict]]:
    features = liquidity_features_wide(ohlcv)
    ic_rows, neutral_rows, regime_rows = [], [], []
    hypo = "C_liquidity_exhaustion"

    for feat_name, signal in features.items():
        suites = apply_neutralization_suite(
            signal, sectors=sectors, spy_ret=spy_ret, close=close
        )
        ics = {}
        for neu_name, neu_sig in suites.items():
            rows = evaluate_feature_panel(
                neu_sig,
                close,
                feature_name=feat_name,
                hypothesis=hypo,
                neutralization=neu_name,
                regime_mask=regime_mask,
                full_metrics=(neu_name == "raw"),
            )
            ic_rows.extend(rows)
            for r in rows:
                if r["horizon"] == 5:
                    ics[neu_name] = r["ic_spearman"]
            if neu_name != "raw":
                for r in rows:
                    if r["horizon"] == 5:
                        regime_rows.append(
                            {
                                "feature": feat_name,
                                "hypothesis": hypo,
                                "neutralization": neu_name,
                                **{k: r[k] for k in r if k.startswith("ic_regime")},
                            }
                        )

        neutral_rows.append(
            {
                "feature": feat_name,
                "hypothesis": hypo,
                "ic_raw_5d": ics.get("raw", np.nan),
                "ic_sector_neutral_5d": ics.get("sector_neutral", np.nan),
                "ic_beta_neutral_5d": ics.get("beta_neutral", np.nan),
                "ic_momentum_neutral_5d": ics.get("momentum_neutral", np.nan),
                "classification": classify_structural_exposure(
                    ics.get("raw", np.nan),
                    ics.get("sector_neutral", np.nan),
                    ics.get("beta_neutral", np.nan),
                    ics.get("momentum_neutral", np.nan),
                ),
            }
        )
    return ic_rows, neutral_rows, regime_rows


def run_hypothesis_b(
    close: pd.DataFrame,
    *,
    sectors: dict[str, str],
    spy_ret: pd.Series,
    regime_mask: pd.Series,
) -> tuple[list[dict], list[dict], list[dict]]:
    features = build_breadth_features_wide(close)
    ic_rows, neutral_rows, regime_rows = [], [], []
    hypo = "B_sector_breadth_deterioration"

    for feat_name, signal in features.items():
        suites = apply_neutralization_suite(
            signal, sectors=sectors, spy_ret=spy_ret, close=close
        )
        ics = {}
        for neu_name, neu_sig in suites.items():
            rows = evaluate_feature_panel(
                neu_sig,
                close,
                feature_name=feat_name,
                hypothesis=hypo,
                neutralization=neu_name,
                regime_mask=regime_mask,
                full_metrics=(neu_name == "raw"),
            )
            ic_rows.extend(rows)
            for r in rows:
                if r["horizon"] == 5:
                    ics[neu_name] = r["ic_spearman"]

        neutral_rows.append(
            {
                "feature": feat_name,
                "hypothesis": hypo,
                "ic_raw_5d": ics.get("raw", np.nan),
                "ic_sector_neutral_5d": ics.get("sector_neutral", np.nan),
                "ic_beta_neutral_5d": ics.get("beta_neutral", np.nan),
                "ic_momentum_neutral_5d": ics.get("momentum_neutral", np.nan),
                "classification": classify_structural_exposure(
                    ics.get("raw", np.nan),
                    ics.get("sector_neutral", np.nan),
                    ics.get("beta_neutral", np.nan),
                    ics.get("momentum_neutral", np.nan),
                ),
            }
        )
        for neu_name in ("sector_neutral", "beta_neutral"):
            sub = [r for r in ic_rows if r["feature"] == feat_name and r["neutralization"] == neu_name]
            for r in sub:
                regime_rows.append(
                    {
                        "feature": feat_name,
                        "hypothesis": hypo,
                        "neutralization": neu_name,
                        "horizon": r["horizon"],
                        "ic_regime_on": r.get("ic_regime_on"),
                        "ic_regime_off": r.get("ic_regime_off"),
                    }
                )
    return ic_rows, neutral_rows, regime_rows


def run_hypothesis_a(
    ohlcv: dict[str, pd.DataFrame],
    close: pd.DataFrame,
    *,
    sectors: dict[str, str],
    spy_ret: pd.Series,
    regime_mask: pd.Series,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    panels, earn_meta = build_earnings_panel(ohlcv)
    ic_rows, neutral_rows, regime_rows = [], [], []
    hypo = "A_post_earnings_drift"

    feat_names = [c for c in next(iter(panels.values())).columns if c != "earn_event_active"]
    for feat_name in feat_names:
        signal = pd.concat({t: panels[t][feat_name] for t in panels}, axis=1)
        suites = apply_neutralization_suite(
            signal, sectors=sectors, spy_ret=spy_ret, close=close
        )
        ics = {}
        for neu_name, neu_sig in suites.items():
            rows = evaluate_feature_panel(
                neu_sig,
                close,
                feature_name=feat_name,
                hypothesis=hypo,
                neutralization=neu_name,
                regime_mask=regime_mask,
                full_metrics=(neu_name == "raw"),
            )
            ic_rows.extend(rows)
            for r in rows:
                if r["horizon"] == 5:
                    ics[neu_name] = r["ic_spearman"]

        neutral_rows.append(
            {
                "feature": feat_name,
                "hypothesis": hypo,
                "ic_raw_5d": ics.get("raw", np.nan),
                "ic_sector_neutral_5d": ics.get("sector_neutral", np.nan),
                "ic_beta_neutral_5d": ics.get("beta_neutral", np.nan),
                "ic_momentum_neutral_5d": ics.get("momentum_neutral", np.nan),
                "classification": classify_structural_exposure(
                    ics.get("raw", np.nan),
                    ics.get("sector_neutral", np.nan),
                    ics.get("beta_neutral", np.nan),
                    ics.get("momentum_neutral", np.nan),
                ),
                "leakage_risk": "HIGH",
            }
        )

    earn_meta["hypothesis"] = hypo
    earn_meta.update(HYPOTHESIS_META[hypo])
    return ic_rows, neutral_rows, regime_rows, earn_meta


def synthesize_verdict(
    neutral_rows: list[dict],
    *,
    ic_threshold: float = 0.02,
) -> dict[str, Any]:
    """Honest falsification summary — failure is valid."""
    surviving = []
    failed = []
    for row in neutral_rows:
        ic = row.get("ic_momentum_neutral_5d") or row.get("ic_beta_neutral_5d") or row.get("ic_raw_5d")
        cls = row.get("classification", "")
        if cls == "potential_independent" and ic is not None and abs(ic) >= ic_threshold:
            surviving.append(row)
        else:
            failed.append(row)

    return {
        "signals_tested": len(neutral_rows),
        "potential_independent_count": len(surviving),
        "failed_or_structural_count": len(failed),
        "true_alpha_claim": len(surviving) > 0,
        "verdict": (
            "NO independent cross-sectional alpha after neutralization"
            if len(surviving) == 0
            else f"{len(surviving)} signal(s) warrant deeper audit (not production)"
        ),
        "surviving_features": [r["feature"] for r in surviving],
    }


def run_all_hypotheses(
    ohlcv: dict[str, pd.DataFrame],
    close: pd.DataFrame,
    spy_close: pd.Series,
    tickers: tuple[str, ...],
) -> dict[str, Any]:
    sectors = sector_series(tickers)
    spy_ret = spy_close.pct_change()
    regime_mask = _spy_regime_mask(spy_close)

    ic_all, neu_all, reg_all = [], [], []
    reports: dict[str, Any] = {"hypotheses": {}}

    ic_a, neu_a, reg_a, meta_a = run_hypothesis_a(
        ohlcv, close, sectors=sectors, spy_ret=spy_ret, regime_mask=regime_mask
    )
    ic_all += ic_a
    neu_all += neu_a
    reg_all += reg_a
    reports["hypotheses"]["A"] = {**HYPOTHESIS_META["A_post_earnings_drift"], **meta_a}

    ic_b, neu_b, reg_b = run_hypothesis_b(
        close, sectors=sectors, spy_ret=spy_ret, regime_mask=regime_mask
    )
    ic_all += ic_b
    neu_all += neu_b
    reg_all += reg_b
    reports["hypotheses"]["B"] = HYPOTHESIS_META["B_sector_breadth_deterioration"]

    ic_c, neu_c, reg_c = run_hypothesis_c(
        ohlcv, close, sectors=sectors, spy_ret=spy_ret, regime_mask=regime_mask
    )
    ic_all += ic_c
    neu_all += neu_c
    reg_all += reg_c
    reports["hypotheses"]["C"] = HYPOTHESIS_META["C_liquidity_exhaustion"]

    reports["falsification"] = synthesize_verdict(neu_all)
    reports["philosophy"] = (
        "Objective is falsification, not alpha manufacturing. "
        "If all hypotheses fail after neutralization, that is a valid outcome."
    )
    return {
        "ic_rows": ic_all,
        "neutral_rows": neu_all,
        "regime_rows": reg_all,
        "hypothesis_report": reports,
    }
