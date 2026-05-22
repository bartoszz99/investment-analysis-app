"""
Breadth research runner — hypothesis-driven; isolated from production stack.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from research.breadth.breadth_features import build_breadth_features
from research.breadth.breadth_tests import (
    decile_curves_for_features,
    hypothesis_a_fragility,
    hypothesis_b_thrust,
    hypothesis_c_reversal,
    run_breadth_feature_ic,
)
from research.breadth.universe_loader import ETF_TARGETS, load_component_closes, load_etf_close
from research.common.feature_neutralization import neutralize_feature_panel
from research.common.signal_evaluation import rolling_ic, spearman_ic
from research.common.forward_returns import forward_return

RESULTS = Path("results")
PLOTS = RESULTS / "plots"
PERIOD = "2y"


def _neutralize_breadth(
    breadth: pd.DataFrame,
    etf: str,
    etf_close: pd.Series,
    sector_closes: pd.DataFrame,
) -> dict[str, pd.Series]:
    spy_ret = sector_closes["SPY"].pct_change()
    mom = etf_close.shift(1) / etf_close.shift(21) - 1.0
    sectors = sector_closes.pct_change()
    sectors = sectors.drop(columns=[c for c in sectors.columns if c == etf], errors="ignore")
    out = {}
    for col in breadth.columns:
        out[col] = neutralize_feature_panel(breadth[col], spy_ret, sectors, mom)
    return out


def _plot_deciles(curves: dict, etf: str) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    for feat, curve in curves.items():
        if curve.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(curve.index.astype(str), curve.values)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_title(f"{etf} — {feat} decile fwd 20d")
        ax.set_xlabel("Decile")
        ax.set_ylabel("Mean forward return")
        fig.tight_layout()
        fig.savefig(PLOTS / f"breadth_deciles_{etf}_{feat}.png", dpi=120)
        plt.close(fig)


def _plot_ic_stability(signal: pd.Series, close: pd.Series, etf: str, name: str) -> None:
    fwd = forward_return(close, 20)
    ic = rolling_ic(signal, fwd, 60)
    if ic.empty:
        return
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(ic.index, ic.values, lw=1)
    ax.axhline(0, color="gray", ls="--")
    ax.set_title(f"{etf} IC stability — {name}")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(PLOTS / f"breadth_ic_stability_{etf}_{name}.png", dpi=120)
    plt.close(fig)


def run_breadth_research(period: str = PERIOD) -> dict:
    print("\n=== ETF Breadth Research ===")
    RESULTS.mkdir(parents=True, exist_ok=True)

    spy = load_etf_close("SPY", period)
    calendar = spy.index
    sector_closes = pd.DataFrame({t: load_etf_close(t, period) for t in ETF_TARGETS}).reindex(calendar)

    all_ic: list[dict] = []
    summaries: dict = {"hypotheses": {}, "etfs": {}, "philosophy_notes": {}}

    for etf in ETF_TARGETS:
        print(f"  Processing {etf}...")
        panel = load_component_closes(etf, period, calendar)
        breadth = build_breadth_features(panel)
        etf_close = load_etf_close(etf, period).reindex(calendar)
        etf_close.name = etf

        neutral = _neutralize_breadth(breadth, etf, etf_close, sector_closes)
        ic_rows = run_breadth_feature_ic(breadth, etf_close, etf, neutral)
        all_ic.extend(ic_rows)

        ha = hypothesis_a_fragility(breadth, etf_close)
        hb = hypothesis_b_thrust(breadth, etf_close)
        hc = hypothesis_c_reversal(breadth, etf_close)
        summaries["hypotheses"][etf] = {"A": ha, "B": hb, "C": hc}
        summaries["etfs"][etf] = {
            "universe_source": panel.source,
            "n_components": len(panel.tickers),
        }

        curves = decile_curves_for_features(
            breadth, etf_close, ["pct_above_sma20", "breadth_thrust", "return_dispersion"]
        )
        _plot_deciles(curves, etf)
        _plot_ic_stability(breadth["pct_above_sma20"], etf_close, etf, "pct_above_sma20")

    ic_df = pd.DataFrame(all_ic)
    ic_df.to_csv(RESULTS / "breadth_ic.csv", index=False)

    # Research philosophy scoring
    best_neutral = ic_df.dropna(subset=["ic_neutral"]).sort_values("ic_neutral", key=abs, ascending=False)
    summaries["top_neutral_ic"] = best_neutral.head(10).to_dict(orient="records") if len(best_neutral) else []
    summaries["philosophy_notes"] = {
        "A_other_side": "Late index buyers without broad participation",
        "B_other_side": "Underweight allocators forced to chase breadth",
        "C_other_side": "Concentrated leadership holders vs broad market",
        "cost_persistence": "Requires edge > daily ETF friction (~2-5bps)",
    }

    with open(RESULTS / "breadth_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summaries, fh, indent=2, default=str)

    print(f"  Saved {RESULTS / 'breadth_ic.csv'}")
    print(f"  Saved {RESULTS / 'breadth_summary.json'}")
    return {"ic": ic_df, "summary": summaries}


if __name__ == "__main__":
    run_breadth_research()
