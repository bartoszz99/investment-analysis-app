"""
Passive flow & market concentration research runner.
Research-only — no strategies, no portfolio optimization.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research.breadth.universe_loader import load_component_closes, load_etf_close
from research.passive_flow.concentration_features import (
    DATA_LIMITATION,
    PASSIVE_FLOW_ETFS,
    build_concentration_features,
)
from research.passive_flow.concentration_tests import (
    event_a_narrow_rally,
    event_b_passive_dominance,
    event_c_participation_recovery,
    run_all_events,
)
from research.passive_flow.distribution_analysis import compare_to_unconditional
from research.passive_flow.passive_flow_summary import build_research_answers, save_summary

RESULTS = Path("results")
PLOTS = RESULTS / "plots"
PERIOD = "10y"

CALENDAR_REGIMES = {
    "2008_2015": ("2008-01-01", "2015-12-31"),
    "2016_2019": ("2016-01-01", "2019-12-31"),
    "2020_2022": ("2020-01-01", "2022-12-31"),
    "2023_2025": ("2023-01-01", "2025-12-31"),
}


def _plot_event_distribution(
    mask: pd.Series,
    close: pd.Series,
    etf: str,
    event: str,
    horizon: int = 20,
) -> None:
    from research.common.forward_returns import forward_return

    PLOTS.mkdir(parents=True, exist_ok=True)
    fwd = forward_return(close, horizon).dropna()
    m = mask.reindex(fwd.index).fillna(False)
    if m.sum() < 5:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(fwd[~m], bins=30, alpha=0.5, label="non-event", density=True)
    ax.hist(fwd[m], bins=15, alpha=0.6, label="event", density=True)
    ax.axvline(0, color="gray", ls="--", lw=0.8)
    ax.set_title(f"{etf} {event} — forward {horizon}d return distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / f"passive_event_dist_{etf}_{event}_{horizon}d.png", dpi=120)
    plt.close(fig)


def _plot_concentration_breadth_scatter(features: pd.DataFrame, etf: str) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    df = features[["concentration_ratio_top10", "breadth_pct_above_sma20"]].dropna()
    if len(df) < 30:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df["concentration_ratio_top10"], df["breadth_pct_above_sma20"], alpha=0.3, s=8)
    ax.set_xlabel("Concentration (top10 weight proxy)")
    ax.set_ylabel("Breadth (% above SMA20)")
    ax.set_title(f"{etf} — concentration vs breadth")
    fig.tight_layout()
    fig.savefig(PLOTS / f"passive_conc_breadth_{etf}.png", dpi=120)
    plt.close(fig)


def _plot_regime_comparison(regime_df: pd.DataFrame, metric: str = "mean_diff") -> None:
    if regime_df.empty:
        return
    PLOTS.mkdir(parents=True, exist_ok=True)
    pivot = regime_df.pivot_table(index="regime", columns="event", values=metric, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8, 4))
    pivot.plot(kind="bar", ax=ax)
    ax.axhline(0, color="gray", ls="--")
    ax.set_title(f"Regime comparison — {metric} (20d fwd vs unconditional)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / f"passive_regime_{metric}.png", dpi=120)
    plt.close(fig)


def _regime_analysis(
    features: pd.DataFrame,
    etf_close: pd.Series,
    etf: str,
) -> list[dict]:
    rows = []
    events = {
        "A_narrow_rally": event_a_narrow_rally(features, etf_close),
        "B_passive_dominance": event_b_passive_dominance(features),
        "C_participation_recovery": event_c_participation_recovery(features),
    }
    for regime_name, (start, end) in CALENDAR_REGIMES.items():
        sl = etf_close.loc[start:end]
        if len(sl) < 30:
            continue
        idx = sl.index
        for ename, mask in events.items():
            sub_mask = mask.reindex(idx).fillna(False)
            cmp = compare_to_unconditional(sub_mask, etf_close.reindex(idx), horizon=20)
            rows.append(
                {
                    "etf": etf,
                    "regime": regime_name,
                    "event": ename,
                    "n_events": int(sub_mask.sum()),
                    "mean_diff": cmp.get("mean_diff"),
                    "left_tail_diff": cmp.get("left_tail_diff"),
                    "correction_prob_diff": cmp.get("correction_prob_diff"),
                }
            )
    return rows


def _vol_regime_analysis(
    features: pd.DataFrame,
    etf_close: pd.Series,
    etf: str,
) -> list[dict]:
    ret = etf_close.pct_change()
    vol = ret.shift(1).rolling(20).std()
    high_vol = vol >= vol.expanding(60).quantile(0.70)
    rows = []
    for label, mask in [("low_vol", ~high_vol), ("high_vol", high_vol)]:
        events = event_a_narrow_rally(features, etf_close) & mask.reindex(features.index).fillna(False)
        cmp = compare_to_unconditional(events, etf_close, 20)
        rows.append(
            {
                "etf": etf,
                "regime": label,
                "event": "A_narrow_rally",
                "n_events": int(events.sum()),
                **{k: cmp.get(k) for k in ("mean_diff", "left_tail_diff", "correction_prob_diff")},
            }
        )
    return rows


def run_passive_flow_research(period: str = PERIOD) -> dict:
    print("\n=== Passive Flow & Concentration Research ===")
    print(f"LIMITATION: {DATA_LIMITATION}\n")
    RESULTS.mkdir(parents=True, exist_ok=True)

    spy = load_etf_close("SPY", period)
    calendar = spy.index

    all_events: list[dict] = []
    dist_rows: list[dict] = []
    regime_rows: list[dict] = []

    for etf in PASSIVE_FLOW_ETFS:
        print(f"  {etf}...")
        panel = load_component_closes(etf, period, calendar)
        etf_close = load_etf_close(etf, period).reindex(calendar)
        features = build_concentration_features(panel, etf_close)

        studies = run_all_events(features, etf_close, etf)
        all_events.extend(studies)

        for s in studies:
            for d in s["distributions"]:
                dist_rows.append({"etf": etf, "event": s["event"], **d})
            cmp = s["vs_unconditional_20d"]
            dist_rows.append(
                {
                    "etf": etf,
                    "event": s["event"],
                    "horizon": 20,
                    "comparison": "vs_unconditional",
                    **{k: v for k, v in cmp.items() if k not in ("event", "unconditional")},
                }
            )

        regime_rows.extend(_regime_analysis(features, etf_close, etf))
        regime_rows.extend(_vol_regime_analysis(features, etf_close, etf))

        _plot_concentration_breadth_scatter(features, etf)
        for ename, fn in [
            ("A_narrow_rally", lambda: event_a_narrow_rally(features, etf_close)),
            ("B_passive_dominance", lambda: event_b_passive_dominance(features)),
            ("C_participation_recovery", lambda: event_c_participation_recovery(features)),
        ]:
            _plot_event_distribution(fn(), etf_close, etf, ename, 20)

    events_df = pd.DataFrame(
        [
            {
                "etf": s["etf"],
                "event": s["event"],
                "n_events": s["n_events"],
                "mean_diff_20d": s["vs_unconditional_20d"].get("mean_diff"),
                "correction_prob_diff_20d": s["vs_unconditional_20d"].get("correction_prob_diff"),
                "left_tail_diff_20d": s["vs_unconditional_20d"].get("left_tail_diff"),
            }
            for s in all_events
        ]
    )
    events_df.to_csv(RESULTS / "passive_flow_events.csv", index=False)

    dist_df = pd.DataFrame(dist_rows)
    dist_df.to_csv(RESULTS / "passive_flow_distribution.csv", index=False)

    regime_df = pd.DataFrame(regime_rows)
    regime_df.to_csv(RESULTS / "passive_flow_regimes.csv", index=False)

    summary = build_research_answers(all_events, regime_rows, DATA_LIMITATION)
    summary["event_studies"] = all_events
    summary["top_events"] = events_df.to_dict(orient="records")
    save_summary(summary, RESULTS / "passive_flow_summary.json")

    _plot_regime_comparison(regime_df, "mean_diff")
    _plot_regime_comparison(regime_df, "correction_prob_diff")

    print(f"  Saved {RESULTS / 'passive_flow_events.csv'}")
    print(f"  Saved {RESULTS / 'passive_flow_summary.json'}")
    print(f"\n  Q1 fragility label: {summary['Q1_concentration_predicts_fragility']['label']}")
    print(f"  Q4 regime verdict:  {summary['Q4_persistent_across_regimes']['verdict']}")

    return {"events": events_df, "distribution": dist_df, "regimes": regime_df, "summary": summary}


if __name__ == "__main__":
    run_passive_flow_research()
