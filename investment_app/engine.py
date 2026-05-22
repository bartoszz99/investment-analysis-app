"""
Analysis engine — USA and Poland (GPW). Self-contained; no research/ imports.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

from investment_app.data.gpw_loader import benchmark_ticker
from investment_app.data.market_region import is_poland, parse_region
from investment_app.data.ticker_mapper import display_ticker, normalize_ticker
from investment_app.explain import build_explanation
from investment_app.models.fundamental import analyze_fundamental
from investment_app.models.poland_structural import analyze_poland_structural
from investment_app.models.structural import analyze_structural
from investment_app.models.technical import analyze_technical
from investment_app.scoring import compute_decision

APP_DIR = Path(__file__).resolve().parent
CACHE_DIR = APP_DIR / "cache"
RESULTS_DIR = APP_DIR / "results"
PERIOD = "2y"

IDEA_MAP = {
    "momentum": "momentum",
    "value": "value",
    "earnings": "earnings",
    "breakout": "breakout",
    "macro": "momentum",
}


@dataclass
class AnalysisRequest:
    ticker: str
    asset_type: str
    idea_type: str
    horizon: str
    market_region: str = "USA"


@dataclass
class AnalysisResult:
    ticker: str
    display_ticker: str
    market_region: str
    asset_type: str
    idea_type: str
    horizon: str
    decision: str
    final_score: float
    breakdown: dict
    fundamental: dict
    technical: dict
    structural: dict
    explanation: dict
    as_of: str
    price_at_analysis: float | None = None


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df.sort_index()


def fetch_prices(ticker: str, period: str = PERIOD, *, region: str = "USA") -> pd.DataFrame:
    sym = normalize_ticker(ticker, region)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{sym}_{period}.csv"
    if cache_path.exists():
        age_h = (
            datetime.now(timezone.utc)
            - datetime.fromtimestamp(cache_path.stat().st_mtime, tz=timezone.utc)
        ).total_seconds() / 3600
        if age_h < 12:
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if len(df) >= 120:
                return df

    raw = yf.Ticker(sym).history(period=period, auto_adjust=True)
    if raw is None or raw.empty:
        raise ValueError(f"No price data for {sym}")
    df = _normalize(raw)
    df.to_csv(cache_path)
    return df


def _fetch_benchmark(region: str, calendar: pd.DatetimeIndex, period: str = PERIOD) -> pd.Series:
    sym = benchmark_ticker(region)
    try:
        df = fetch_prices(sym, period, region=region)
        return df["Close"].reindex(calendar).ffill()
    except Exception:
        if is_poland(region):
            df = fetch_prices("SPY", period, region="USA")
            return df["Close"].reindex(calendar).ffill()
        raise


def analyze(
    request: AnalysisRequest,
    *,
    user_note: str | None = None,
    save_journal: bool = True,
) -> AnalysisResult:
    region = parse_region(request.market_region)
    sym = normalize_ticker(request.ticker, region)
    disp = display_ticker(sym)
    idea_key = IDEA_MAP.get(request.idea_type.lower(), "momentum")

    ohlcv = fetch_prices(sym, region=region)
    close = ohlcv["Close"]
    volume = ohlcv["Volume"] if "Volume" in ohlcv.columns else close * 0
    bench = _fetch_benchmark(region, close.index)

    try:
        info = yf.Ticker(sym).info
    except Exception:
        info = {}

    fundamental = analyze_fundamental(disp, info)
    technical = analyze_technical(close, bench, idea=idea_key)

    if is_poland(region):
        structural = analyze_poland_structural(
            close,
            volume,
            ticker=sym,
            fundamental_score=fundamental["score_fundamental"],
        )
        liq = structural.get("liquidity", {}).get("liquidity_risk")
        spec = structural.get("speculation", {}).get("speculation_risk")
        decision = compute_decision(
            fundamental["score_fundamental"],
            technical["score_technical"],
            structural["score_structural"],
            structural["classification"],
            regime_stability=structural.get("regime_stability", 0.7),
            region=region,
            liquidity_risk=liq,
            speculation_risk=spec,
        )
    else:
        structural = analyze_structural(close, bench, idea=idea_key)
        decision = compute_decision(
            fundamental["score_fundamental"],
            technical["score_technical"],
            structural["score_structural"],
            structural["classification"],
            regime_stability=structural.get("regime_stability", 1.0),
            region=region,
        )
        liq = spec = None

    explanation = build_explanation(
        disp,
        request.idea_type,
        request.horizon,
        request.asset_type,
        fundamental,
        technical,
        structural,
        decision,
        market_region=region,
    )

    price_at = float(close.iloc[-1]) if len(close) else None

    result = AnalysisResult(
        ticker=sym,
        display_ticker=disp,
        market_region=region,
        asset_type=request.asset_type,
        idea_type=request.idea_type,
        horizon=request.horizon,
        decision=decision["decision"],
        final_score=decision["final_score"],
        breakdown=decision["breakdown"],
        fundamental=fundamental,
        technical=technical,
        structural=structural,
        explanation=explanation,
        as_of=datetime.now(timezone.utc).isoformat(),
        price_at_analysis=price_at,
    )

    _save_result(result)
    if save_journal:
        from investment_app.journal.store import save_analysis_to_journal

        save_analysis_to_journal(result, price_at_analysis=price_at, user_note=user_note)
    return result


def _save_result(result: AnalysisResult) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{result.display_ticker}_{result.idea_type}_{result.market_region}_latest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2, default=str)
