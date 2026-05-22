"""
Liquid large-cap universe — Nasdaq 100 (default) or S&P 100 subset.
No microcaps, OTC, or illiquid names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

UniverseName = Literal["nasdaq100", "sp100"]

# Nasdaq-100 style liquid large caps (research subset; expandable via fetch)
NASDAQ_100_TICKERS: tuple[str, ...] = (
    "AAPL", "MSFT", "AMZN", "NVDA", "META", "GOOGL", "GOOG", "TSLA", "AVGO", "COST",
    "NFLX", "AMD", "PEP", "ADBE", "CSCO", "TMUS", "CMCSA", "INTC", "TXN", "QCOM",
    "AMGN", "HON", "INTU", "AMAT", "ISRG", "BKNG", "VRTX", "ADP", "SBUX", "GILD",
    "ADI", "REGN", "PANW", "MU", "LRCX", "MDLZ", "KLAC", "SNPS", "CDNS", "MELI",
    "ASML", "PYPL", "CRWD", "MAR", "ORLY", "ABNB", "FTNT", "CHTR", "MNST", "ADSK",
    "CTAS", "WDAY", "PCAR", "NXPI", "MRVL", "PAYX", "AZN", "CPRT", "ROST", "KDP",
    "EA", "LULU", "DASH", "CSX", "FAST", "VRSK", "EXC", "BKR", "DXCM", "BIIB",
    "IDXX", "XEL", "KHC", "GEHC", "ZS", "ON", "CDW", "FANG", "DLTR", "TTWO",
    "MCHP", "ODFL", "CTSH", "WBD", "TEAM", "DDOG", "ARM", "SMCI",
    "COIN", "MSTR", "PDD", "JD", "BIDU", "LCID", "RIVN", "SNOW",
    "PLTR", "APP", "CEG", "GFS", "MDB", "OKTA",
)

def _dedupe(seq: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for t in seq:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return tuple(out)


NASDAQ_100_UNIQUE: tuple[str, ...] = _dedupe(NASDAQ_100_TICKERS)

SP100_TICKERS: tuple[str, ...] = (
    "AAPL", "MSFT", "AMZN", "NVDA", "BRK-B", "GOOGL", "META", "JPM", "UNH", "XOM",
    "JNJ", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "COST", "PEP",
    "KO", "WMT", "BAC", "CRM", "TMO", "ACN", "MCD", "CSCO", "ABT", "LIN",
    "DHR", "ADBE", "NKE", "TXN", "PM", "NEE", "DIS", "WFC", "RTX", "INTC",
    "AMD", "QCOM", "IBM", "GE", "CAT", "GS", "MS", "BLK", "AXP", "SPGI",
    "LOW", "UNP", "HON", "AMGN", "DE", "SBUX", "GILD", "MDT", "BMY", "PLD",
    "ISRG", "SYK", "VRTX", "REGN", "LMT", "BA", "UPS", "T", "CMCSA", "ORCL",
    "INTU", "NOW", "PFE", "ELV", "CI", "CB", "MMC", "SO", "DUK", "SLB",
    "EOG", "COP", "MO", "BDX", "ZTS", "ADI", "PANW", "KLAC", "SNPS", "CDNS",
    "AMAT", "LRCX", "MU", "BKNG", "MAR", "TGT", "CVS", "USB", "PNC", "TFC",
)

# GICS-style coarse sector buckets for neutralization / breadth
SECTOR_MAP: dict[str, str] = {
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology", "AMD": "Technology",
    "INTC": "Technology", "QCOM": "Technology", "AVGO": "Technology", "TXN": "Technology",
    "ADBE": "Technology", "CRM": "Technology", "ORCL": "Technology", "CSCO": "Technology",
    "INTU": "Technology", "AMAT": "Technology", "LRCX": "Technology", "KLAC": "Technology",
    "SNPS": "Technology", "CDNS": "Technology", "PANW": "Technology", "CRWD": "Technology",
    "FTNT": "Technology", "ZS": "Technology", "DDOG": "Technology", "MDB": "Technology",
    "SNOW": "Technology", "PLTR": "Technology", "ARM": "Technology", "SMCI": "Technology",
    "NOW": "Technology", "IBM": "Technology", "ACN": "Technology", "CTSH": "Technology",
    "GOOGL": "Communication", "GOOG": "Communication", "META": "Communication",
    "NFLX": "Communication", "CMCSA": "Communication", "CHTR": "Communication",
    "TMUS": "Communication", "T": "Communication", "DIS": "Communication", "WBD": "Communication",
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary", "HD": "Consumer Discretionary",
    "LOW": "Consumer Discretionary", "NKE": "Consumer Discretionary", "SBUX": "Consumer Discretionary",
    "MCD": "Consumer Discretionary", "BKNG": "Consumer Discretionary", "MAR": "Consumer Discretionary",
    "ABNB": "Consumer Discretionary", "ORLY": "Consumer Discretionary", "ROST": "Consumer Discretionary",
    "LULU": "Consumer Discretionary", "TGT": "Consumer Discretionary", "DASH": "Consumer Discretionary",
    "PEP": "Consumer Staples", "KO": "Consumer Staples", "PG": "Consumer Staples", "COST": "Consumer Staples",
    "WMT": "Consumer Staples", "MDLZ": "Consumer Staples", "KDP": "Consumer Staples", "KHC": "Consumer Staples",
    "MNST": "Consumer Staples", "PM": "Consumer Staples", "MO": "Consumer Staples", "EL": "Consumer Staples",
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials", "GS": "Financials", "MS": "Financials",
    "BLK": "Financials", "AXP": "Financials", "USB": "Financials", "PNC": "Financials", "TFC": "Financials",
    "CB": "Financials", "MMC": "Financials", "SPGI": "Financials", "V": "Financials", "MA": "Financials",
    "PYPL": "Financials", "COIN": "Financials", "MSTR": "Financials",
    "UNH": "Health Care", "JNJ": "Health Care", "MRK": "Health Care", "ABBV": "Health Care",
    "PFE": "Health Care", "LLY": "Health Care", "TMO": "Health Care", "ABT": "Health Care",
    "DHR": "Health Care", "BMY": "Health Care", "AMGN": "Health Care", "GILD": "Health Care",
    "VRTX": "Health Care", "REGN": "Health Care", "ISRG": "Health Care", "SYK": "Health Care",
    "MDT": "Health Care", "BDX": "Health Care", "ZTS": "Health Care", "BIIB": "Health Care",
    "DXCM": "Health Care", "IDXX": "Health Care", "AZN": "Health Care",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "EOG": "Energy", "SLB": "Energy",
    "FANG": "Energy", "BKR": "Energy",
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities", "EXC": "Utilities", "XEL": "Utilities",
    "CEG": "Utilities",
    "LIN": "Materials", "CAT": "Materials", "DE": "Industrials", "HON": "Industrials",
    "UPS": "Industrials", "UNP": "Industrials", "RTX": "Industrials", "LMT": "Industrials",
    "BA": "Industrials", "GE": "Industrials", "CSX": "Industrials", "ODFL": "Industrials",
    "FAST": "Industrials", "CPRT": "Industrials", "CTAS": "Industrials", "PCAR": "Industrials",
    "PLD": "Real Estate",
    "ASML": "Technology", "MELI": "Consumer Discretionary", "PDD": "Consumer Discretionary",
    "JD": "Consumer Discretionary", "BIDU": "Communication", "APP": "Communication",
}

DEFAULT_SECTOR = "Other"


@dataclass(frozen=True)
class UniverseSpec:
    name: UniverseName
    tickers: tuple[str, ...]
    max_size: int


def get_sector(ticker: str) -> str:
    return SECTOR_MAP.get(ticker.upper(), DEFAULT_SECTOR)


def get_universe(
    name: UniverseName = "nasdaq100",
    *,
    max_size: int = 100,
    min_tickers: int = 30,
) -> UniverseSpec:
    pool = NASDAQ_100_UNIQUE if name == "nasdaq100" else SP100_TICKERS
    tickers = tuple(pool[:max_size])
    if len(tickers) < min_tickers:
        raise ValueError(f"Universe {name} has only {len(tickers)} tickers (min {min_tickers})")
    return UniverseSpec(name=name, tickers=tickers, max_size=max_size)


def sector_series(tickers: tuple[str, ...]) -> dict[str, str]:
    return {t: get_sector(t) for t in tickers}
