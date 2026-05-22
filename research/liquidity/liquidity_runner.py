"""
Liquidity exhaustion research runner — isolated from production.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

from research.breadth.universe_loader import ETF_TARGETS, load_etf_close
from research.common.feature_neutralization import neutralize_feature_panel
from research.common.forward_returns import forward_return
from research.common.signal_evaluation import rolling_ic
from research.liquidity.liquidity_features import build_liquidity_features
from research.liquidity.liquidity_tests import (
    hypothesis_d_mean_reversion,
    hypothesis_e_weak_followthrough,
    run_liquidity_feature_ic,
)

RESULTS = Path("results")
PLOTS = RESULTS / "plots"
PERIOD = "2y"


def _load_ohlcv(ticker: str, period: str) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period=period)
    if df.index.tz:
        df.index = df.index.tz_localize(None)
    return df[["Open", "High", "Low", "Close", "Volume"]].sort_index()


def _plot_conditional_returns(
    signal: pd.Series,
    close: pd.Series,
    etf: str,
    label: str,
) -> None:
    fwd = forward_return(close, 5)
    aligned = pd.concat([signal.rename("s"), fwd.rename("f")], axis=1).dropna()
    if len(aligned) < 30:
        return
    q = pd.qcut(aligned["s"].rank(method="first"), 5, labels=False)
    cond = aligned.groupby(q)["f"].mean()
    cum = (1 + cond).cumprod()

    PLOTS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(cond.index.astype(str), cond.values)
    axes[0].set_title(f"{etf} {label} — conditional 5d fwd")
    axes[1].plot(cum.index, cum.values, marker="o")
    axes[1].set_title("Cumulative conditional returns")
    fig.tight_layout()
    fig.savefig(PLOTS / f"liquidity_conditional_{etf}_{label}.png", dpi=120)
    plt.close(fig)


def run_liquidity_research(period: str = PERIOD) -> dict:
    print("\n=== Liquidity Exhaustion Research ===")
    RESULTS.mkdir(parents=True, exist_ok=True)

    sector_closes = pd.DataFrame({t: load_etf_close(t, period) for t in ETF_TARGETS})
    calendar = sector_closes.index
    spy_ret = sector_closes["SPY"].pct_change()

    all_ic: list[dict] = []
    summaries: dict = {"hypotheses": {}, "etfs": {}}

    for etf in ETF_TARGETS:
        print(f"  Processing {etf}...")
        ohlcv = _load_ohlcv(etf, period).reindex(calendar)
        close = ohlcv["Close"]
        features = build_liquidity_features(ohlcv)

        mom = close.shift(1) / close.shift(21) - 1.0
        sectors = sector_closes.pct_change().drop(columns=[etf], errors="ignore")
        neutral = {
            c: neutralize_feature_panel(features[c], spy_ret, sectors, mom)
            for c in features.columns
        }

        ic_rows = run_liquidity_feature_ic(features, close, etf, neutral)
        all_ic.extend(ic_rows)

        hd = hypothesis_d_mean_reversion(features, close, etf)
        he = hypothesis_e_weak_followthrough(features, close, etf)
        summaries["hypotheses"][etf] = {"D": hd, "E": he}

        _plot_conditional_returns(features["exhaustion_composite"], close, etf, "exhaustion")
        ic_roll = rolling_ic(features["exhaustion_composite"], forward_return(close, 20), 60)
        if not ic_roll.empty:
            PLOTS.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(ic_roll.index, ic_roll.values)
            ax.axhline(0, color="gray", ls="--")
            ax.set_title(f"{etf} exhaustion IC stability")
            fig.tight_layout()
            fig.savefig(PLOTS / f"liquidity_ic_stability_{etf}.png", dpi=120)
            plt.close(fig)

    ic_df = pd.DataFrame(all_ic)
    ic_df.to_csv(RESULTS / "liquidity_ic.csv", index=False)

    best = ic_df.dropna(subset=["ic_neutral"]).sort_values("ic_neutral", key=abs, ascending=False)
    summaries["top_neutral_ic"] = best.head(10).to_dict(orient="records") if len(best) else []
    summaries["philosophy_notes"] = {
        "D_other_side": "Late momentum entrants / forced rebalancers",
        "E_other_side": "Trapped volume at failed breakout",
        "friction": "Mean reversion must exceed ETF round-trip costs",
    }

    with open(RESULTS / "liquidity_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summaries, fh, indent=2, default=str)

    print(f"  Saved {RESULTS / 'liquidity_ic.csv'}")
    print(f"  Saved {RESULTS / 'liquidity_summary.json'}")
    return {"ic": ic_df, "summary": summaries}


if __name__ == "__main__":
    run_liquidity_research()
