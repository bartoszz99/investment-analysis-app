"""
Watchlist persistence — local JSON file.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from investment_app.engine import APP_DIR

WATCHLIST_DIR = APP_DIR / "watchlist"
WATCHLIST_FILE = WATCHLIST_DIR / "watchlist.json"


def _load_raw() -> dict:
    WATCHLIST_DIR.mkdir(parents=True, exist_ok=True)
    if not WATCHLIST_FILE.exists():
        return {"items": []}
    with open(WATCHLIST_FILE, encoding="utf-8") as f:
        return json.load(f)


def _save_raw(data: dict) -> None:
    WATCHLIST_DIR.mkdir(parents=True, exist_ok=True)
    with open(WATCHLIST_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def list_items() -> list[dict]:
    return list(_load_raw().get("items", []))


def add_item(
    ticker: str,
    *,
    asset_type: str = "Stock",
    idea_type: str = "momentum",
    horizon: str = "medium",
    market_region: str = "USA",
) -> None:
    from investment_app.data.ticker_mapper import normalize_ticker

    ticker = normalize_ticker(ticker, market_region)
    data = _load_raw()
    items = data["items"]
    if any(i["ticker"] == ticker for i in items):
        return
    items.append(
        {
            "ticker": ticker,
            "asset_type": asset_type,
            "idea_type": idea_type,
            "horizon": horizon,
            "market_region": market_region,
            "added_at": datetime.now(timezone.utc).isoformat(),
            "last_score": None,
            "prev_score": None,
            "last_decision": None,
            "prev_decision": None,
            "last_structural": None,
            "last_trend": None,
            "last_updated": None,
            "risk_flags": [],
        }
    )
    _save_raw(data)


def remove_item(ticker: str) -> None:
    ticker = ticker.upper().strip()
    data = _load_raw()
    data["items"] = [i for i in data["items"] if i["ticker"] != ticker]
    _save_raw(data)


def update_item_snapshot(ticker: str, snapshot: dict) -> None:
    data = _load_raw()
    for item in data["items"]:
        if item["ticker"] == ticker.upper():
            item["prev_score"] = item.get("last_score")
            item["prev_decision"] = item.get("last_decision")
            item.update(snapshot)
            item["last_updated"] = datetime.now(timezone.utc).isoformat()
            break
    _save_raw(data)
