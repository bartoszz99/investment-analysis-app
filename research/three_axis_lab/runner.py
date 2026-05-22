"""
3-axis idea evaluation runner.

Run: python -m research.three_axis_lab.runner

Answers: "Does this investment idea hang together as a hypothesis?"
Does NOT answer: "Will this make money?"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

from research.three_axis_lab.axis_fundamental import score_fundamental
from research.three_axis_lab.axis_structural import score_structural
from research.three_axis_lab.axis_technical import score_technical
from research.three_axis_lab.idea_scoring import combine_scores

RESULTS = Path("results")
PERIOD = "2y"

# ETFs + liquid equities (expand as needed)
DEFAULT_TICKERS: tuple[str, ...] = (
    "SPY", "QQQ", "IWM", "XLK", "XLF", "XLE",
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "UNH", "XOM",
)

IDEA_TYPES = ("momentum", "breakout", "mean_reversion", "earnings_reaction")

SECTOR_ETF_MAP = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Energy": "XLE",
    "Healthcare": "SPY",
    "Consumer Cyclical": "SPY",
    "Consumer Defensive": "SPY",
}


def _log(msg: str) -> None:
    print(msg, flush=True)
    sys.stdout.flush()


def _load_history(ticker: str, period: str) -> pd.DataFrame | None:
    try:
        df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        if df is None or df.empty:
            return None
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df[["Open", "High", "Low", "Close", "Volume"]].sort_index()
    except Exception:
        return None


def _sector_etf(sector: str) -> str | None:
    return SECTOR_ETF_MAP.get(sector)


def run_three_axis_lab(
    tickers: tuple[str, ...] = DEFAULT_TICKERS,
    *,
    period: str = PERIOD,
    ideas: tuple[str, ...] = ("momentum",),
) -> dict:
    _log("\n=== 3-Axis Investment Idea Evaluation ===")
    _log("Question: is the hypothesis coherent? (not: does it alpha?)")

    spy_df = _load_history("SPY", period)
    if spy_df is None:
        raise RuntimeError("SPY data required")
    spy_close = spy_df["Close"]

    axis_rows: list[dict] = []
    neutral_rows: list[dict] = []
    stability_rows: list[dict] = []
    classifications: list[dict] = []

    for ticker in tickers:
        ohlcv = _load_history(ticker, period)
        if ohlcv is None or len(ohlcv) < 252:
            _log(f"  skip {ticker}: insufficient data")
            continue

        close, volume = ohlcv["Close"], ohlcv["Volume"]
        try:
            info = yf.Ticker(ticker).info
        except Exception:
            info = {}

        fund = score_fundamental(ticker, info)
        sector = fund.get("sector", "unknown")
        sec_etf = _sector_etf(sector) if isinstance(sector, str) else None
        sec_df = _load_history(sec_etf, period) if sec_etf else None
        sec_close = sec_df["Close"] if sec_df is not None else None

        for idea in ideas:
            tech = score_technical(close, idea=idea)
            struct = score_structural(
                ticker,
                close,
                volume,
                idea=idea,
                spy_close=spy_close,
                sector_close=sec_close,
            )
            combined = combine_scores(
                fund["score_fundamental"],
                tech["score_technical"],
                struct["score_structural"],
                structural_class=struct["structural_class"],
            )

            row = {
                "ticker": ticker,
                "idea": idea,
                "score_fundamental": fund["score_fundamental"],
                "score_technical": tech["score_technical"],
                "score_structural": struct["score_structural"],
                "idea_score": combined["idea_score"],
                "directional_score": combined["directional_score"],
                "verdict": combined["verdict"],
                "structural_class": struct["structural_class"],
                "quality_trend": fund.get("quality_trend"),
                "sector": sector,
            }
            axis_rows.append(row)

            neutral_rows.append(
                {
                    "ticker": ticker,
                    "idea": idea,
                    "ic_mean": struct["ic_mean"],
                    "residual_ic_mean": struct["residual_ic_mean"],
                    "ic_1d": struct["ic_1d"],
                    "ic_5d": struct["ic_5d"],
                    "ic_20d": struct["ic_20d"],
                    "structural_class": struct["structural_class"],
                    "reasons": "; ".join(struct["reasons"]),
                }
            )

            for regime, ric in struct.get("regime_ic", {}).items():
                stability_rows.append(
                    {
                        "ticker": ticker,
                        "idea": idea,
                        "regime": regime,
                        "ic_5d": ric,
                    }
                )

            classifications.append(
                {
                    "ticker": ticker,
                    "idea": idea,
                    **combined,
                    "fundamental": fund,
                    "technical": tech,
                    "structural": struct,
                    "philosophy": (
                        "Coherent hypothesis requires mechanism + stable structure; "
                        "not backtest PnL."
                    ),
                }
            )

        _log(f"  {ticker} done ({len(ideas)} idea(s))")

    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(axis_rows).to_csv(RESULTS / "axis_scores.csv", index=False)
    pd.DataFrame(neutral_rows).to_csv(RESULTS / "neutralization_report.csv", index=False)
    pd.DataFrame(stability_rows).to_csv(RESULTS / "stability_report.csv", index=False)

    summary = {
        "n_evaluations": len(axis_rows),
        "tickers": list(tickers),
        "ideas": list(ideas),
        "period": period,
        "verdict_counts": pd.Series([r["verdict"] for r in axis_rows]).value_counts().to_dict()
        if axis_rows
        else {},
        "philosophy": (
            "System classifies investment hypothesis coherence, not alpha. "
            "MARKET_EXPOSURE = idea is mostly beta/sector/momentum. "
            "NOISE = no structural support."
        ),
    }
    with open(RESULTS / "idea_classification.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "evaluations": classifications}, f, indent=2, default=str)

    _log("\n--- Summary ---")
    _log(json.dumps(summary["verdict_counts"], indent=2))
    _log("\nArtifacts:")
    for name in (
        "axis_scores.csv",
        "idea_classification.json",
        "neutralization_report.csv",
        "stability_report.csv",
    ):
        _log(f"  {RESULTS / name}")

    return {"summary": summary, "evaluations": classifications}


if __name__ == "__main__":
    run_three_axis_lab(ideas=IDEA_TYPES)
