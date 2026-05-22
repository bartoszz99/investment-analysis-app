"""
Decision layer runner — BUY / WATCH / IGNORE from 3-axis lab artifacts.

Prerequisite: python -m research.three_axis_lab.runner

Run: python -m research.decision_layer.runner
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from research.decision_layer.counterparty_model import assess_counterparty
from research.decision_layer.decision_engine import decide
from research.decision_layer.explanation_builder import build_explanation
from research.decision_layer.input_schema import (
    AxisLabSnapshot,
    InvestmentIdea,
    SIGNAL_TO_LAB_IDEA,
)
from research.decision_layer.risk_filter import apply_risk_filters

RESULTS = Path("results")
ETF_TICKERS = frozenset({"SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "VTI", "VOO"})


def _log(msg: str) -> None:
    print(msg, flush=True)
    sys.stdout.flush()


def _regime_stability(stability_df: pd.DataFrame, ticker: str, idea: str) -> float:
    sub = stability_df[(stability_df["ticker"] == ticker) & (stability_df["idea"] == idea)]
    if sub.empty:
        return 0.5
    ics = sub["ic_5d"].dropna()
    if len(ics) < 2:
        return 0.5
    signs = np.sign(ics.replace(0, np.nan).dropna())
    if len(signs) < 2:
        return 0.5
    same = (signs > 0).all() or (signs < 0).all()
    return 1.0 if same else max(0.2, 1.0 - float(signs.std()))



def load_lab_snapshots(results_dir: Path = RESULTS) -> dict[tuple[str, str], AxisLabSnapshot]:
    axis_path = results_dir / "axis_scores.csv"
    neu_path = results_dir / "neutralization_report.csv"
    stab_path = results_dir / "stability_report.csv"

    for p in (axis_path, neu_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}. Run: python -m research.three_axis_lab.runner"
            )

    axis = pd.read_csv(axis_path)
    neu = pd.read_csv(neu_path)
    stab = pd.read_csv(stab_path) if stab_path.exists() else pd.DataFrame()

    lookup: dict[tuple[str, str], AxisLabSnapshot] = {}
    for _, row in axis.iterrows():
        ticker = str(row["ticker"]).upper()
        idea = str(row["idea"])
        neu_row = neu[(neu["ticker"] == ticker) & (neu["idea"] == idea)]
        nr = neu_row.iloc[0] if len(neu_row) else None
        lookup[(ticker, idea)] = AxisLabSnapshot(
            score_fundamental=float(row["score_fundamental"]),
            score_technical=float(row["score_technical"]),
            score_structural=float(row["score_structural"]),
            structural_class=str(row.get("structural_class", "NOISE")),
            neutralization_result=str(nr["structural_class"]) if nr is not None else "UNKNOWN",
            ic_mean=float(nr["ic_mean"]) if nr is not None else np.nan,
            residual_ic_mean=float(nr["residual_ic_mean"]) if nr is not None else np.nan,
            regime_stability=_regime_stability(stab, ticker, idea),
            verdict=str(row.get("verdict", "")),
        )
    return lookup


def ideas_from_axis_scores(lookup: dict) -> list[InvestmentIdea]:
    ideas = []
    for ticker, lab_idea in lookup:
        sig = next((k for k, v in SIGNAL_TO_LAB_IDEA.items() if v == lab_idea), "momentum")
        universe = "ETF" if ticker in ETF_TICKERS else "EQUITY"
        ideas.append(
            InvestmentIdea(
                ticker=ticker,
                universe=universe,  # type: ignore[arg-type]
                signal_type=sig,  # type: ignore[arg-type]
                time_horizon="medium",
            )
        )
    return ideas


def evaluate_idea(
    idea: InvestmentIdea,
    lookup: dict[tuple[str, str], AxisLabSnapshot],
) -> dict | None:
    key = idea.key()
    lab = lookup.get(key)
    if lab is None:
        return None

    cp = assess_counterparty(idea)
    risk = apply_risk_filters(lab, cp)
    result = decide(lab, cp, risk)
    return build_explanation(idea, lab, cp, risk, result)


def run_decision_layer(
    ideas: list[InvestmentIdea] | None = None,
    *,
    results_dir: Path = RESULTS,
) -> dict:
    _log("\n=== Decision Layer (BUY / WATCH / IGNORE) ===")
    _log("Reads 3-axis lab outputs — does not run research.")

    lookup = load_lab_snapshots(results_dir)
    if ideas is None:
        ideas = ideas_from_axis_scores(lookup)

    decisions: list[dict] = []
    skipped: list[str] = []

    for idea in ideas:
        out = evaluate_idea(idea, lookup)
        if out is None:
            skipped.append(f"{idea.ticker}/{idea.signal_type}")
            continue
        decisions.append(out)

    payload = {
        "philosophy": (
            "Answers: is this a coherent investment decision vs market exposure? "
            "Does NOT answer: will it profit?"
        ),
        "thresholds": {"BUY": 0.65, "WATCH": 0.40},
        "n_decisions": len(decisions),
        "skipped_missing_lab": skipped,
        "decisions": decisions,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "decisions.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    counts = pd.Series([d["decision"] for d in decisions]).value_counts().to_dict()
    _log(f"\nDecisions: {counts}")
    _log(f"Written: {out_path}")
    if skipped:
        _log(f"Skipped (no lab row): {len(skipped)}")

    return payload


if __name__ == "__main__":
    run_decision_layer()
