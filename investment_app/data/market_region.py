"""
Stałe regionów rynkowych.
"""

from __future__ import annotations

from enum import Enum


class MarketRegion(str, Enum):
    USA = "USA"
    POLAND = "POLAND"


REGIONS = (MarketRegion.USA.value, MarketRegion.POLAND.value)

REGION_LABELS = {
    MarketRegion.USA.value: "Stany Zjednoczone",
    MarketRegion.POLAND.value: "Polska (GPW)",
}


def parse_region(value: str | None) -> str:
    if not value:
        return MarketRegion.USA.value
    v = value.upper().strip()
    if v in ("PL", "GPW", "POLAND", "POL"):
        return MarketRegion.POLAND.value
    return MarketRegion.USA.value


def is_poland(region: str) -> bool:
    return parse_region(region) == MarketRegion.POLAND.value
