"""
GPW price loading — thin wrapper over yfinance with .WA symbols.
"""

from __future__ import annotations

from investment_app.data.ticker_mapper import GPW_SEED_TICKERS, normalize_ticker
from investment_app.data.market_region import MarketRegion

GPW_BENCHMARK = "WIG.WA"  # fallback: use SPY trend only if missing
GPW_BENCHMARK_FALLBACK = "SPY"


def benchmark_ticker(region: str) -> str:
    if region == MarketRegion.POLAND.value:
        return GPW_BENCHMARK
    return "SPY"


def seed_ticker_list() -> list[str]:
    return list(GPW_SEED_TICKERS)


def prepare_request_ticker(ticker: str, region: str) -> str:
    return normalize_ticker(ticker, region)
