"""
Ticker normalization — GPW symbols use .WA on Yahoo Finance.
"""

from __future__ import annotations

from investment_app.data.market_region import MarketRegion, is_poland, parse_region

GPW_SUFFIX = ".WA"

# Example / common GPW tickers (also accepts bare symbols)
GPW_SEED_TICKERS: tuple[str, ...] = (
    "CDR.WA",
    "PKN.WA",
    "PKO.WA",
    "ALE.WA",
    "DNP.WA",
)

# State-linked or concentrated ownership heuristics (flags only)
STATE_OR_CONCENTRATED: dict[str, list[str]] = {
    "PKO": ["state-linked bank", "policy-sensitive"],
    "PKN": ["state-controlled energy", "regulated pricing risk"],
    "PZU": ["state-linked insurer"],
    "KGH": ["state-linked miner"],
    "PEO": ["state-linked bank"],
    "ORL": ["state-linked refiner"],
}


def strip_suffix(ticker: str) -> str:
    t = ticker.upper().strip()
    if t.endswith(GPW_SUFFIX):
        return t[: -len(GPW_SUFFIX)]
    return t


def normalize_ticker(ticker: str, region: str) -> str:
    """
    USA: uppercase symbol as-is (AAPL).
    Poland: append .WA if missing (CDR → CDR.WA).
    """
    t = ticker.upper().strip().replace(" ", "")
    if is_poland(region):
        base = strip_suffix(t)
        return f"{base}{GPW_SUFFIX}"
    return strip_suffix(t)


def display_ticker(ticker: str) -> str:
    """Human-friendly symbol (CDR.WA → CDR)."""
    return strip_suffix(ticker)


def region_badge(region: str) -> str:
    return "GPW" if is_poland(region) else "USA"


def ownership_flags(ticker: str) -> list[str]:
    base = strip_suffix(ticker)
    return list(STATE_OR_CONCENTRATED.get(base, []))


def infer_region_from_ticker(ticker: str) -> str:
    if ticker.upper().strip().endswith(GPW_SUFFIX):
        return MarketRegion.POLAND.value
    return MarketRegion.USA.value
